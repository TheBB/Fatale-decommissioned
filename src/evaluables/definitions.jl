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
struct GlobalCoords{N,T,L} <: Evaluable{CoordsType{N,T,L}}
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


"""
    GetProperty{T,S}

Function accessing a field named S of type T.
"""
struct GetProperty{T,S,A} <: Evaluable{T}
    arg :: A
    GetProperty{T,S}(arg::A) where {T,S,A} = new{T,S,A}(arg)
end

arguments(self::GetProperty) = [self.arg]

@generated (::GetProperty{T,S})(element, quadpt, arg) where {T,S} = :(arg.$S)
