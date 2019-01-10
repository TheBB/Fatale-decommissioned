module Domains

using LinearAlgebra
using StaticArrays
using ..Transforms
using ..Elements
using ..Evaluables

export TensorDomain
export basis, Lagrange


abstract type Domain{Elt,Ref,N} <: AbstractArray{Elt,N} end

# Basis types
abstract type Basis end
struct Lagrange <: Basis end


struct TensorDomain{D} <: Domain{FullElement{D,Shift{D,Float64}}, Tensor{D,NTuple{D,Simplex{1}}}, D}
    size :: NTuple{D,Int}
    TensorDomain(I::Int...) = new{length(I)}(I)
end

@inline Base.size(self::TensorDomain) = self.size
@inline Base.IndexStyle(::Type{TensorDomain}) = IndexCartesian()
@inline function Base.getindex(self::TensorDomain{D}, I::Vararg{Int,D}) where {D}
    @boundscheck checkbounds(self, I...)
    shift = SVector{D,Float64}(I) - 1.0
    FullElement(Shift(shift))
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

    outer
end

end
