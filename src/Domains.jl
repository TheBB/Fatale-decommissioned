module Domains

using StaticArrays
using ..Transforms
using ..Elements

export TensorDomain


abstract type Domain{Elt,Ref,N} <: AbstractArray{Elt,N} end


struct TensorDomain{D} <: Domain{FullElement{D,Shift{D,Float64}}, Tensor{D,NTuple{D,Simplex{1}}}, D}
    size :: NTuple{D,Int}
end

TensorDomain(I::Int...) = TensorDomain{length(I)}(I)

@inline Base.size(self::TensorDomain) = self.size
@inline Base.IndexStyle(::Type{TensorDomain}) = IndexCartesian()
@inline function Base.getindex(self::TensorDomain{D}, I::Vararg{Int,D}) where {D}
    @boundscheck checkbounds(self, I...)
    shift = SVector{D,Float64}(I) - 1.0
    FullElement(Shift(shift))
end

end
