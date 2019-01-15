struct Block
    indices :: Tupl
    data :: Evaluable

    function Block(indices, data)
        @assert length(indices) == ndims(data)
        @assert all(length(ind) == size(data, i) for (i, ind) in enumerate(indices))
        new(indices, data)
    end
end

function blocks(self::Evaluable)
    indices = Tupl((Constant(SVector{n}(1:n)) for n in size(self))...)
    [Block(indices, self)]
end

function blocks(self::Inflate)
    # TODO: Lift this limitation
    @assert !(self.data isa Inflate)

    indices = [
        i == self.axis ? self.indices : Constant(SVector(1:s...))
        for (i, s) in enumerate(size(self))
    ]
    [Block(Tupl(indices...), self.data)]
end


struct CompiledBlock{I,D}
    indices :: I
    data :: D
end

@inline indices(self::CompiledBlock, element) = self.indices(element, nothing)
@inline (self::CompiledBlock)(element, quadpt) = self.data(element, quadpt)

compile(self::Block) = CompiledBlock(optimize(self.indices), optimize(self.data))


struct CompiledBlocks{K<:Tuple}
    blocks :: K
end

@inline Base.getindex(self::CompiledBlocks, i) = self.blocks[i]
@inline Base.length(self::CompiledBlocks) = length(self.blocks)

compile(self::Evaluable) = CompiledBlocks(Tuple(compile(block) for block in blocks(self)))


# =============================================================================
# Optimized evaluables that are directly callable

struct OptimizedEvaluable{T, I, K} <: Evaluable{T}
    funcs :: K
    OptimizedEvaluable{T,I}(funcs::K) where {T,I,K} = new{T,I,K}(funcs)
end

function optimize(self::Evaluable{T}) where {T}
    sequence = linearize(self)
    funcs = Tuple(stage.func for stage in sequence)
    Indices = Tuple{(Tuple{stage.arginds...} for stage in sequence)...}
    OptimizedEvaluable{T,Indices}(funcs)
end

@generated function (self::OptimizedEvaluable{T,I,K})(element, quadpt) where {T,I,K}
    nfuncs = length(K.parameters)
    syms = [gensym() for _ in 1:nfuncs]
    argsyms = [[syms[j] for j in tp.parameters] for tp in I.parameters]

    codes = Expr[]
    for (i, (sym, args)) in enumerate(zip(syms, argsyms))
        code = codegen(K.parameters[i], :(self.funcs[$i]), :element, :quadpt, args...)
        push!(codes, :($sym = $code))
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
