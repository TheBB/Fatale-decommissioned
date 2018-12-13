"""
    CompiledFunction{T} <: Evaluable{T}

Represents a compiled evaluable.
"""
struct CompiledFunction{T} <: Evaluable{T}
    mod :: Module
    func :: Function
end


"""
    (::CompiledFunction)(element::Element, quadpt::SVector)

Evaluate a compiled evaluable at a certain element and quadrature point.
"""
@inline function (self::CompiledFunction{T})(element::Element, quadpt::SVector) where {T}
    self.func(element, quadpt) :: T
end


"""
    compile(::Evaluable) :: CompiledFunction

Compile an evaluable and return a compiled function that can be evaluated.
"""
function compile(self::Evaluable{T}) where {T}
    bodycode = codegen(self)
    funccode = quote
        function evaluate(element::Element, quadpt::SVector)
            $bodycode
        end
    end

    mod = Module()
    Core.eval(mod, :(using StaticArrays))
    Core.eval(mod, :(using Fatale.Transforms))
    Core.eval(mod, :(using Fatale.Elements))
    Core.eval(mod, funccode)

    CompiledFunction{T}(mod, mod.evaluate)
end
