
"""
    CompiledFunction{T} <: Evaluable{T}

Represents a compiled evaluable.
"""
struct CompiledFunction{T} <: Evaluable{T}
    mod :: Module
    func :: Function
end


"""
    (::CompiledFunction)() :: CompiledEvaluator

Prepare an evaluation function by allocating temporary storage.
"""
@inline function (self::CompiledFunction{T})() where {T}
    Evaluator{T}(self.func())
end


"""
    Evaluator{T}

Represents a compiled evaluator (a closure with allocated storage.)
"""
struct Evaluator{T}
    func :: Function
end


"""
    (::Evaluator)(element::Element, quadpt::SVector)

Evaluate a compiled evaluator at a certain element and quadrature point.
"""
@inline function (self::Evaluator{T})(element::Element, quadpt::SVector) where {T}
    self.func(element, quadpt) :: T
end


"""
    compile(::Evaluable) :: CompiledFunction

Compile an evaluable and return a compiled function that can be evaluated.
"""
function compile(self::Evaluable{T}) where {T}
    sequence = linearize(self)

    # Code for allocating storage
    storage_code = Expr[]
    for data in sequence
        append!(storage_code, [:($sym = $expr) for (sym, expr) in data.storage_exprs])
    end

    # Code for evaluation
    eval_code = Expr[]
    for data in sequence
        code = codegen(data.func; data.storage_symbols...)
        push!(eval_code, :($(data.tgtsym) = $code))
    end

    funccode = quote
        function mkevaluate()
            $(storage_code...)
            function evaluate(element::Element, quadpt::SVector)
                $(eval_code...)
            end
        end
    end

    mod = Module()
    Core.eval(mod, :(using StaticArrays))
    Core.eval(mod, :(using Fatale.Transforms))
    Core.eval(mod, :(using Fatale.Elements))
    Core.eval(mod, funccode)

    CompiledFunction{T}(mod, mod.mkevaluate)
end


# =============================================================================
# For internal use by the compiler

mutable struct Stage
    func :: Evaluable
    tgtsym :: Symbol
    storage_exprs :: Vector{Tuple{Symbol,Expr}}
    storage_symbols :: Vector{Tuple{Symbol,Symbol}}
    arginds :: Vector{Int}

    function Stage(func::Evaluable)
        stage = new(func, gensym())

        st = storage(func)
        newsyms = [gensym() for _ in st]
        stage.storage_exprs = collect(zip(newsyms, values(st)))
        stage.storage_symbols = collect(zip(keys(st), newsyms))

        stage
    end
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
