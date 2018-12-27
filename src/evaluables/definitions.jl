# Return type of LocalCoords and GlobalCoords
const CoordsType{N,T,L} = NamedTuple{
    (:point, :grad),
    Tuple{MVector{N,T}, MMatrix{N,N,T,L}}
}

"""
    LocalCoords(N, T)

Function returning the local (reference) `N`-dimensional coordinates of the
quadrature point, with element type `T`.
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
    GlobalCoords(N, T)

Function returning the global (physical) `N`-dimensional coordinates of the
quadrature point, with element type `T`.
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
    GetProperty{T, S}(arg)

Function accessing a field of `arg` named `S` of type `T`.
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


"""
    Monomials(arg, degree)

Computes all monomials of the input up to `degree`, returning an array of size

    (size(arg)..., degree+1)
"""
struct Monomials{D, In, Out} <: Evaluable{Out}
    arg :: Evaluable{In}

    function Monomials(arg::ArrayEvaluable, degree::Int)
        newsize = (size(arg)..., degree + 1)
        new{degree, restype(arg), marray(newsize, eltype(arg))}(arg)
    end
end

arguments(self::Monomials) = [self.arg]

storage(self::Monomials) = restype(self)(undef)

@generated function (self::Monomials{D})(_, _, st, arg) where {D}
    colons = [Colon() for _ in 1:ndims(self)-1]
    codes = [:(st[$(colons...), $(i+1)] .= st[$(colons...), $i] .* arg) for i in 1:D]
    quote
        st[$(colons...), 1] .= 1
        $(codes...)
        st
    end
end


"""
    Constant(v)

A function returning the constant value `v`.
"""
struct Constant{T} <: Evaluable{T}
    value :: T
end

@inline (self::Constant)(_, _) = self.value


"""
    Contract(l, r, linds, rinds, tinds)

Represents a fully unrolled tensor contraction operation, in Einstein notation
as

    result[tinds...] = l[linds...] * r[rinds...]

where `tinds`, `linds` and `rinds` are tuples of integers with no "gaps", i.e.
all integers from 1 through to the maximum must be used.
"""
struct Contract{Linds, Rinds, Tinds, InL, InR, Out} <: Evaluable{Out}
    lft :: Evaluable{InL}
    rgt :: Evaluable{InR}

    function Contract(lft::ArrayEvaluable, rgt::ArrayEvaluable, linds, rinds, tinds)
        @assert length(linds) == ndims(lft)
        @assert length(rinds) == ndims(rgt)

        dims = Dict(
            (k => v for (k, v) in zip(linds, size(lft)))...,
            (k => v for (k, v) in zip(rinds, size(rgt)))...,
        )
        @assert all(size(lft,i) == dims[linds[i]] for i in 1:ndims(lft))
        @assert all(size(rgt,i) == dims[rinds[i]] for i in 1:ndims(rgt))

        tsize = Tuple(dims[tind] for tind in tinds)
        Out = marray(tsize, promote_type(eltype(lft), eltype(rgt)))
        new{linds, rinds, tinds, restype(lft), restype(rgt), Out}(lft, rgt)
    end
end

arguments(self::Contract) = [self.lft, self.rgt]

storage(self::Contract) = restype(self)(undef)

@generated function (self::Contract{linds, rinds, tinds})(_, _, st, lft, rgt) where {linds, rinds, tinds}
    max_axis = max(linds..., rinds..., tinds...)
    sizes = zeros(Int, max_axis)
    for (k, v) in flatten((zip(linds, size(lft)), zip(rinds, size(rgt)), zip(tinds, size(self))))
        sizes[k] = v
    end

    codes = Expr[]
    for indices in product((1:n for n in sizes)...)
        li = indices[collect(linds)]
        ri = indices[collect(rinds)]
        ti = indices[collect(tinds)]
        push!(codes, :(st[$(ti...)] += lft[$(li...)] * rgt[$(ri...)]))
    end

    quote
        st .= $(zero(eltype(self)))
        $(codes...)
        st
    end
end
