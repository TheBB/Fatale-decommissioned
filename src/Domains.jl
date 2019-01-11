module Domains

using LinearAlgebra
using StaticArrays
using ..Transforms
using ..Elements
using ..Evaluables

export TensorDomain, TensorDofMap
export basis, Lagrange


# Basis types

abstract type Basis end
struct Lagrange <: Basis end


# Degree-of-Freedom maps

abstract type DofMap{T,D} <: AbstractArray{T,D} end

struct TensorDofMap{T,D,L} <: DofMap{T,D}

    # The index of the first basis function in each element, as
    # understood by separate one-dimensional bases.
    roots :: NTuple{D,Vector{Int}}

    # The "jump" between two adjacent basis functions for each
    # dimension.
    strides :: NTuple{D,Int}

    # size: the number of elements in each dimension
    # steps: the stride for each dimension
    # locsize: the number of basis functions per element for each dimension
    function TensorDofMap(size::NTuple{D,Int}, steps::NTuple{D,Int}, locsize::NTuple{D,Int}) where {D}
        roots = Tuple(collect(range(0, step=st, length=sz)) for (sz, st) in zip(size, steps))
        nfuncs = Tuple(lsz + (sz - 1) * st for (sz, st, lsz) in zip(size, steps, locsize))
        strides = Tuple(prod(nfuncs[1:k-1]) for k in 1:D)
        rtype = SVector{prod(locsize), Int}
        new{rtype, D, Tuple{locsize...}}(roots, strides)
    end
end

@inline Base.size(self::TensorDofMap) = Tuple(length(r) for r in self.roots)
@inline Base.IndexStyle(::Type{<:TensorDofMap}) = IndexCartesian()
@inline Base.getindex(self::TensorDofMap, I::Vararg{Int,D}) where {D} = self[I]

@generated function Base.getindex(self::TensorDofMap{T,D,L}, I::NTuple{D,Int}) where {T,D,L}
    len = length(T)
    temptype = MArray{L, Int, D, len}

    # Compute all the ranges to add into the temp array
    ranges = Expr[]
    for k in 1:D
        ll = L.parameters[k]
        addtype = SArray{Tuple{ones(Int,k-1)..., ll}, Int, k, ll}
        addexpr = :($addtype($(0:ll-1...)))
        push!(ranges, :(temp .+= self.strides[$k] .* (self.roots[$k][I[$k]] .+ $addexpr)))
    end

    quote
        Base.@_inline_meta
        # TODO: Figure out how to properly elide the check with @inbounds
        # This seems to be more involved than I thought...
        # @boundscheck checkbounds(self, I...)
        temp = zero($temptype)
        $(ranges...)
        $T(temp) + 1
    end
end


# Domains

abstract type Domain{Elt,Ref,N} <: AbstractArray{Elt,N} end


const TensorElement{D} = FullElement{D, NTuple{D, Int}, Shift{D, Float64}}
const TensorReference{D} = Tensor{D, NTuple{D, Simplex{1}}}

struct TensorDomain{D} <: Domain{TensorElement{D}, TensorReference{D}, D}
    size :: NTuple{D,Int}
    TensorDomain(I::Int...) = new{length(I)}(I)
end

@inline Base.size(self::TensorDomain) = self.size
@inline Base.IndexStyle(::Type{<:TensorDomain}) = IndexCartesian()
@inline function Base.getindex(self::TensorDomain{D}, I::Vararg{Int,D}) where {D}
    @boundscheck checkbounds(self, I...)
    shift = SVector{D,Float64}(I) - 1.0
    FullElement(Shift(shift), I)
end

function basis(self::TensorDomain{D}, ::Type{Lagrange}, degree) where {D}
    # Generate N single-dimensional Lagrangian bases of the right degree
    poly = Monomials(localpoint(D), degree)
    coeffs = inv(range(0, 1, length=degree+1) .^ reshape(0:degree, 1, :))
    coeffs = SMatrix{degree+1, degree+1}(coeffs)
    basis1d = Contract(poly, Constant(coeffs), (1, 2), (2, 3), (1, 3))

    # Reshape and form an outer product
    factors = [reshape(basis1d[k,:], ones(Int,k-1)..., :) for k in 1:D]
    outer = .*(factors...)

    dofmap = TensorDofMap(size(self), ntuple(_->degree, D), ntuple(_->degree+1, D))

    (reshape(outer, :), Elementwise(dofmap))
end

end
