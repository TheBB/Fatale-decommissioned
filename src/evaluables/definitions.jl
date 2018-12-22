# Return type of LocalCoords and GlobalCoords
const CoordsType{N,T,L} = NamedTuple{
    (:point, :grad),
    Tuple{MVector{N,T}, MMatrix{N,N,T,L}}
}

"""
    LocalCoords{N, T}

Function returning the local (reference) coordinates of the quadrature point.
"""
struct LocalCoords{N,T,L} <: Evaluable{CoordsType{N,T,L}}
    LocalCoords{N,T}() where {N,T} = new{N,T,N*N}()
end

_storage(::LocalCoords{N,T}) where {N,T} = (
    point = MVector{N,T}(undef),
    grad = MMatrix{N,N,T}(undef),
)

function evaluate(::LocalCoords{N}, element, quadpt::SVector{M}, storage) where {M,N}
    storage.mine.point[1:M] .= quadpt
    storage.mine.grad .= Matrix{Float64}(I, N, N)
    apply!(dimtrans(element), storage.mine.point, storage.mine.grad)
    storage.mine
end


"""
    GlobalCoords{N, T}

Function returning the global (physical) coordinates of the quadrature point.
"""
struct GlobalCoords{N,T,L} <: Evaluable{CoordsType}
    GlobalCoords{N,T}() where {N,T} = new{N,T,N*N}()
end

_storage(::GlobalCoords{N,T}) where {N,T} = (
    point = MVector{N,T}(undef),
    grad = MMatrix{N,N,T}(undef),
)

function evaluate(::GlobalCoords{N}, element, quadpt::SVector{M}, storage) where {M,N}
    storage.mine.point[1:M] .= quadpt
    storage.mine.grad .= Matrix{Float64}(I, N, N)
    apply!(globtrans(element), storage.mine.point, storage.mine.grad)
    storage.mine
end
