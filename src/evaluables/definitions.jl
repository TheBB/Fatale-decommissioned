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
    LocalCoords(N) = new{N, Float64, N*N}()
    LocalCoords(N, T) = new{N, T, N*N}()
end

storage(::LocalCoords{N,T}) where {N,T} = (
    point = MVector{N,T}(undef),
    grad = MMatrix{N,N,T}(undef),
)

function (::LocalCoords{N})(element, quadpt::SVector{M}, st) where {M,N}
    st.point[1:M] .= quadpt
    st.grad .= Matrix{Float64}(I, N, N)
    apply!(dimtrans(element), st.point, st.grad)
    st
end


"""
    GlobalCoords{N, T}

Function returning the global (physical) coordinates of the quadrature point.
"""
struct GlobalCoords{N,T,L} <: Evaluable{CoordsType}
    GlobalCoords(N) = new{N, Float64, N*N}()
    GlobalCoords(N, T) = new{N, T, N*N}()
end

storage(::GlobalCoords{N,T}) where {N,T} = (
    point = MVector{N,T}(undef),
    grad = MMatrix{N,N,T}(undef),
)

function (::GlobalCoords{N})(element, quadpt::SVector{M}, st) where {M,N}
    st.point[1:M] .= quadpt
    st.grad .= Matrix{Float64}(I, N, N)
    apply!(globtrans(element), st.point, st.grad)
    st
end
