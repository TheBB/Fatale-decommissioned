module Elements

using FastGaussQuadrature
using StaticArrays
using ..Utils
using ..Transforms

import Base.Iterators: product
import ..Transforms: fromdims, todims

export Simplex, Tensor
export Element, FullElement, SubElement
export quadrule, loctrans, globtrans


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


abstract type Element{D} end

Base.ndims(::Type{<:Element{D}}) where {D} = D
Base.ndims(::Element{D}) where {D} = D

struct FullElement{D, I, Trf<:Transform} <: Element{D}
    transform :: Trf
    index :: I
end

# Convenience constructor that doesn't care about index. This is
# useful for testing evaluables that also don't care about index.
FullElement(trf) = FullElement(trf, nothing)

@generated function FullElement(trf, index)
    @assert trf <: Transform
    @assert fromdims(trf) == todims(trf)
    quote
        $(Expr(:meta, :inline))
        FullElement{$(todims(trf)), $index, $trf}(trf, index)
    end
end

struct SubElement{D, I, Trf<:Transform, Parent<:Element} <: Element{D}
    transform :: Trf
    parent :: Parent
    index :: I
end

# Convenience constructor that doesn't care about index. This is
# useful for testing evaluables that also don't care about index.
SubElement(trf, parent) = SubElement(trf, parent, nothing)

@generated function SubElement(trf, parent, index)
    @assert trf <: Transform
    @assert todims(trf) == ndims(parent)
    quote
        $(Expr(:meta, :inline))
        SubElement{$(todims(trf)), $index, $trf, $parent}(trf, parent, index)
    end
end

@inline loctrans(::FullElement{D}) where {D} = Empty{D,Float64}()
@inline loctrans(self::SubElement) = Chain(self.transform, loctrans(self.parent))

@inline globtrans(self::FullElement) = self.transform
@inline globtrans(self::SubElement) = globtrans(self.parent)

end
