module Transforms

import Base.Iterators: flatten
import Base: @_inline_meta

using LinearAlgebra
using StaticArrays

export Transform
export Empty, Chain, Updim, Shift
export apply


"""
    abstract type Transform{M, N, R} end

A `Transform` represents a transformation from M-dimensional to N-dimensional
space with an element type R.
"""
abstract type Transform{From, To, R<:Real} end

@inline fromdims(::Type{<:Transform{From, To, R}}) where {From, To, R} = From
@inline todims(::Type{<:Transform{From, To, R}}) where {From, To, R} = To
@inline eltype(::Type{<:Transform{From, To, R}}) where {From, To, R} = R
@inline fromdims(::Transform{From, To, R}) where {From, To, R} = From
@inline todims(::Transform{From, To, R}) where {From, To, R} = To
@inline eltype(::Transform{From, To, R}) where {From, To, R} = R


"""
    Empty{D,R}()

A D-dimensional transform that does nothing.
"""
struct Empty{D, R<:Real} <: Transform{D, D, R} end

(::Empty)(point) = point
(::Empty)(point, grad) = (point, grad)


"""
    Chain(transforms...)

Construct single chain transformation from a sequence of transformations to
apply in order.
"""
struct Chain{K<:Tuple{Vararg{Transform}}, From, To, R<:Real} <: Transform{From, To, R}
    chain :: K
end

# Generated constructor with compile-time type assertions
@generated function Chain(trfs...)
    from = fromdims(trfs[1])
    to = todims(trfs[end])
    R = eltype(trfs[1])
    @assert all(R == eltype(trf) for trf in trfs)
    @assert all(todims(prev) == fromdims(next)
                for (prev, next) in zip(trfs[1:end-1], trfs[2:end]))
    quote
        @_inline_meta
        Chain{Tuple{$(trfs...)}, $from, $to, $R}(trfs)
    end
end

@generated function (self::Chain{K})(point) where {K}
    nentries = length(K.parameters)
    codes = [:(point = self.chain[$i](point)) for i in 1:nentries]
    quote
        @_inline_meta
        $(codes...)
        point
    end
end

@generated function (self::Chain{K})(point, grad) where {K}
    nentries = length(K.parameters)
    codes = [:((point, grad) = self.chain[$i](point, grad)) for i in 1:nentries]
    quote
        @_inline_meta
        $(codes...)
        (point, grad)
    end
end


"""
    Updim{Ins, To}(value)

Create a transformation increasing the dimension by one, by inserting an element `value` at index
`Ins` in each input vector. The final result should have dimension `To`.
"""
struct Updim{Ins, From, To, R<:Real} <: Transform{From, To, R}
    data :: R

    @inline function Updim{Ins, To}(data) where {Ins, To}
        @assert 1 <= Ins <= To
        @assert 1 <= To <= 3
        new{Ins, To-1, To, typeof(data)}(data)
    end
end

@generated function (self::Updim{Ins,From})(point) where {Ins,From}
    elements = [:(point[$i]) for i in 1:From]
    insert!(elements, Ins, :(self.data))
    quote
        @_inline_meta
        @SVector [$(elements...)]
    end
end

@generated function (self::Updim{Ins,From})(point, grad) where {Ins,From}
    To = From + 1
    R = eltype(self)

    src_cols = [[:(grad[$i,$j]) for i in 1:From] for j in 1:From]
    for col in src_cols
        insert!(col, Ins, :(zero($R)))
    end

    if To == 1
        new_col = [:(one($R))]
    elseif To == 2
        ((a, b),) = src_cols
        new_col = [b, :(-$a)]
    elseif To == 3
        ((a, c, e), (b, d, f)) = src_cols
        new_col = [:($c*$f - $e*$d), :($e*$b - $a*$f), :($a*$d - $c*$b)]
    end

    elements = flatten((src_cols..., new_col))

    quote
        @_inline_meta
        newgrad = SMatrix{$To,$To}($(elements...))
        (self(point), newgrad)
    end
end


"""
    Shift(x::SVector)

Create a shifting transformation that adds `x` to each input vector.
"""
struct Shift{D,R} <: Transform{D,D,R}
    data :: SVector{D,R}
end

@inline (self::Shift)(point) = point + self.data
@inline (self::Shift)(point, grad) = (point + self.data, grad)


end
