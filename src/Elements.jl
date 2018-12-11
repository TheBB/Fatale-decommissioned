module Elements

using FastGaussQuadrature
using StaticArrays
using ..Utils

import Base.Iterators: product

export Simplex, Tensor
export quadrule


abstract type ReferenceElement{D} end

Base.ndims(::Type{<:ReferenceElement{D}}) where {D} = D
Base.ndims(::ReferenceElement{D}) where {D} = D


"""
    Simplex{D}

Represents a D-dimensional simplex reference element (triangle, tetrahedron etc.)
"""
struct Simplex{D} <: ReferenceElement{D} end

function quadrule(::Simplex{1}, npts::Int)
    (pts, wts) = gausslegendre(npts)
    rwts = wts ./ 2
    rpts = SVector{1,Float64}[SVector((pt+1)/2) for pt in pts]
    (rpts, rwts)
end


"""
    Tensor{<:Tuple{Vararg{ReferenceElement}}}

Represents a tensor product reference element, e.g. for D-dimensional structured
meshes use `Tensor(ntuple(_->Simplex{1}(), D))`.
"""
struct Tensor{D,K<:Tuple{Vararg{ReferenceElement}}} <: ReferenceElement{D}
    terms :: K
end

@generated function Tensor(terms::ReferenceElement...)
    D = sum(ndims(T) for T in terms)
    K = :(Tuple{$(terms...)})
    :(Tensor{$D,$K}(terms))
end

function quadrule(self::Tensor{D}, npts::Int) where {D}
    (pts, wts) = zip((quadrule(term, npts) for term in self.terms)...)
    rwts = vec(outer(wts...))
    rpts = vec(SVector{D, Float64}[SVector(vcat(p...)) for p in product(pts...)])
    (rpts, rwts)
end

end
