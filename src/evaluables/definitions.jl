"""
    LocalCoords{N, T} <: VectorEvaluable{N, T}

Function returning the local (reference) coordinates of the quadrature point.
"""
struct LocalCoords{N,T} <: VectorEvaluable{N,T} end

_storage(::LocalCoords{N,T}) where {N,T} = MVector{N,T}(undef)

function evaluate(::LocalCoords{N,T}, element, quadpt::SVector{M}, storage) where {M,N,T}
    storage.mine[1:M] .= quadpt
    apply!(dimtrans(element), storage.mine)
    storage.mine
end


"""
    GlobalCoords{N, T} <: VectorEvaluable{N, T}

Function returning the global (physical) coordinates of the quadrature point.
"""
struct GlobalCoords{N,T} <: VectorEvaluable{N,T} end

_storage(::GlobalCoords{N,T}) where {N,T} = MVector{N,T}(undef)

function evaluate(::GlobalCoords{N,T}, element, quadpt::SVector{M}, storage) where {M,N,T}
    storage.mine[1:M] .= quadpt
    apply!(globtrans(element), storage.mine)
    storage.mine
end
