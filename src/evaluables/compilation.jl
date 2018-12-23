struct CompiledEvaluable{T, I, K} <: Evaluable{T}
    funcs :: K

    CompiledEvaluable{T,I}(funcs::K) where {T,I,K} = new{T,I,K}(funcs)
end

function compile(self::Evaluable{T}) where {T}
    sequence = linearize(self)
    funcs = Tuple(stage.func for stage in sequence)
    Indices = Tuple{(Tuple{stage.arginds...} for stage in sequence)...}
    CompiledEvaluable{T,Indices}(funcs)
end

storage(self::CompiledEvaluable) = Tuple(storage(func) for func in self.funcs)

(self::CompiledEvaluable)(element, quadpt) = self(element, quadpt, storage(self))

@generated function (self::CompiledEvaluable{T,I,K})(element, quadpt, storage) where {T,I,K}
    nfuncs = length(K.parameters)
    syms = [gensym() for _ in 1:nfuncs]
    argsyms = [[syms[j] for j in tp.parameters] for tp in I.parameters]

    codes = Expr[]
    for (i, (sym, args)) in enumerate(zip(syms, argsyms))
        if storage.parameters[i] == Nothing
            push!(codes, :($sym = self.funcs[$i](
                element, quadpt, $(args...)
            )))
        else
            push!(codes, :($sym = self.funcs[$i](
                element, quadpt, storage[$i], $(args...)
            )))
        end
    end

    quote
        $(codes...)
        $(syms[end])
    end
end


# =============================================================================
# For internal use by the compiler

mutable struct Stage
    func :: Evaluable
    tgtsym :: Symbol
    arginds :: Vector{Int}

    Stage(func::Evaluable) = new(func, gensym())
end

function linearize!(indices, self::Evaluable)
    haskey(indices, self) && return
    for func in arguments(self)
        linearize!(indices, func)
    end
    indices[self] = length(indices) + 1
end

function linearize(self::Evaluable)
    indices = OrderedDict{Evaluable,Int}()
    linearize!(indices, self)
    sequence = [Stage(func) for func in keys(indices)]
    for data in sequence
        data.arginds = Int[indices[arg] for arg in arguments(data.func)]
    end
    sequence
end
