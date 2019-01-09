# Return type of LocalCoords and GlobalCoords
const CoordsType{N,T,L} = NamedTuple{
    (:point, :grad),
    Tuple{SVector{N,T}, SMatrix{N,N,T,L}}
}

"""
    LocalCoords(N, T)

Function returning the local (reference) `N`-dimensional coordinates of the
quadrature point, with element type `T`.
"""
struct LocalCoords{T} <: Evaluable{T}
    LocalCoords(N) = new{CoordsType{N, Float64, N*N}}()
    LocalCoords(N, T) = new{CoordsType{N, T, N*N}}()
end

@inline function (::LocalCoords{<:CoordsType{N,T}})(element, quadpt) where {N,T}
    igrad = SMatrix{N,N,T}(I)
    (point, grad) = loctrans(element)(quadpt, igrad)
    (point=point, grad=grad)
end


"""
    GlobalCoords(N, T)

Function returning the global (physical) `N`-dimensional coordinates of the
quadrature point, with element type `T`.
"""
struct GlobalCoords{T} <: Evaluable{T}
    GlobalCoords(N) = new{CoordsType{N, Float64, N*N}}()
    GlobalCoords(N, T) = new{CoordsType{N, T, N*N}}()
end

arguments(::GlobalCoords{<:CoordsType{N,T}}) where {N,T} = [LocalCoords(N,T)]

@inline function (::GlobalCoords)(element, _, loc)
    (point, grad) = globtrans(element)(loc.point, loc.grad)
    (point=point, grad=grad)
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
struct Monomials{D, T} <: Evaluable{T}
    arg :: Evaluable
    storage :: T

    function Monomials(arg::Evaluable, degree::Int)
        newsize = (size(arg)..., degree + 1)
        rtype = marray(newsize, eltype(arg))
        new{degree, rtype}(arg, rtype(undef))
    end
end

arguments(self::Monomials) = [self.arg]

@generated function (self::Monomials{D})(_, _, arg) where {D}
    colons = [Colon() for _ in 1:ndims(self)-1]
    codes = [:(self.storage[$(colons...), $(i+1)] .= self.storage[$(colons...), $i] .* arg) for i in 1:D]
    quote
        @_inline_meta
        self.storage[$(colons...), 1] .= 1
        $(codes...)
        self.storage
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

# Constants should not be considered equal to each other unless they
# have the same value.
Base.hash(self::Constant, x::UInt64) = hash(self.value, x)
Base.:(==)(l::Constant, r::Constant) = l.value == r.value


"""
    Contract(l, r, linds, rinds, tinds)

Represents a fully unrolled tensor contraction operation, in Einstein notation
as

    result[tinds...] = l[linds...] * r[rinds...]

where `tinds`, `linds` and `rinds` are tuples of integers with no "gaps", i.e.
all integers from 1 through to the maximum must be used.
"""
struct Contract{Linds, Rinds, Tinds, T} <: Evaluable{T}
    lft :: Evaluable
    rgt :: Evaluable
    storage :: T

    function Contract(lft::Evaluable, rgt::Evaluable, linds, rinds, tinds)
        @assert length(linds) == ndims(lft)
        @assert length(rinds) == ndims(rgt)

        dims = Dict(
            (k => v for (k, v) in zip(linds, size(lft)))...,
            (k => v for (k, v) in zip(rinds, size(rgt)))...,
        )
        @assert all(size(lft,i) == dims[linds[i]] for i in 1:ndims(lft))
        @assert all(size(rgt,i) == dims[rinds[i]] for i in 1:ndims(rgt))

        tsize = Tuple(dims[tind] for tind in tinds)
        rtype = marray(tsize, promote_type(eltype(lft), eltype(rgt)))
        new{linds, rinds, tinds, rtype}(lft, rgt, rtype(undef))
    end
end

arguments(self::Contract) = [self.lft, self.rgt]

@generated function (self::Contract{linds, rinds, tinds})(_, _, lft, rgt) where {linds, rinds, tinds}
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
        push!(codes, :(self.storage[$(ti...)] += lft[$(li...)] * rgt[$(ri...)]))
    end

    quote
        @_inline_meta
        self.storage .= $(zero(eltype(self)))
        $(codes...)
        self.storage
    end
end


"""
    GetIndex(arg, inds...)

Represents an expression arg[inds...]. Works with statically known
integers and colons only.
"""
struct GetIndex{Inds, T} <: Evaluable{T}
    arg :: Evaluable
    storage :: T

    function GetIndex(arg::Evaluable, inds...)
        @assert length(inds) == ndims(restype(arg))
        @assert all(
            isa(i, Colon) || isa(i, Integer) && 1 <= i <= s
            for (i, s) in zip(inds, size(arg))
        )

        ressize = Tuple(s for (i, s) in zip(inds, size(arg)) if isa(i, Colon))
        rtype = array(ressize, eltype(arg), arraytype(arg))
        new{Tuple{inds...}, rtype}(arg, rtype(undef))
    end
end

arguments(self::GetIndex) = [self.arg]

@generated function (self::GetIndex{inds})(_, _, arg) where {inds}
    quote
        @_inline_meta
        arg[$(inds.parameters...)]
    end
end
