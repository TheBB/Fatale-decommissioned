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

@inline function (::LocalCoords{N})(element, quadpt::SVector{M}, st) where {M,N}
    st.point[1:M] .= quadpt
    st.grad .= SMatrix{N,N,Float64}(I)
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

arguments(::GlobalCoords{N,T}) where {N,T} = [LocalCoords(N,T)]

storage(::GlobalCoords{N,T}) where {N,T} = (
    point = MVector{N,T}(undef),
    grad = MMatrix{N,N,T}(undef),
)

@inline function (::GlobalCoords)(element, _, st, loc)
    st.point .= loc.point
    st.grad .= loc.grad
    apply!(globtrans(element), st.point, st.grad)
    st
end


"""
    GetProperty{T, S}

Function accessing a field named S of type T.
"""
struct GetProperty{T,S,A} <: Evaluable{T}
    arg :: A
    GetProperty{T,S}(arg::A) where {T,S,A} = new{T,S,A}(arg)
end

arguments(self::GetProperty) = [self.arg]

@generated function (::GetProperty{T,S})(_, _, arg) where {T,S}
    quote
        @_inline_meta
        arg.$S
    end
end
