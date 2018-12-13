"""
    LocalCoords{N, T} <: VectorEvaluable{N, T}

Function returning the local (reference) coordinates of the quadrature point.
"""
struct LocalCoords{N, T} <: VectorEvaluable{N, T} end
codegen(::LocalCoords) = :(apply(dimtrans(element), quadpt))


"""
    GlobalCoords{N, T} <: VectorEvaluable{N, T}

Function returning the global (physical) coordinates of the quadrature point.
"""
struct GlobalCoords{N, T} <: VectorEvaluable{N, T} end
codegen(::GlobalCoords) = :(apply(globtrans(element), quadpt))
