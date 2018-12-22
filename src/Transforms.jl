module Transforms

using LinearAlgebra
using StaticArrays

export Transform
export Empty, Chain, Updim, Shift
export apply!


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
codegen(::Type{<:Empty}, trf, rest...) = :()


"""
    Chain(transforms...)

Construct single chain transformation from a sequence of transformations to
apply in order.
"""
struct Chain{K<:Tuple{Vararg{Transform}}, From, To, R<:Real} <: Transform{From, To, R}
    chain :: K
end

@generated function Chain(trfs...)
    from = fromdims(trfs[1])
    to = todims(trfs[end])
    R = eltype(trfs[1])
    @assert all(R == eltype(trf) for trf in trfs)
    @assert all(todims(prev) == fromdims(next)
                for (prev, next) in zip(trfs[1:end-1], trfs[2:end]))
    quote
        $(Expr(:meta, :inline))
        Chain{Tuple{$(trfs...)}, $from, $to, $R}(trfs)
    end
end

function codegen(::Type{<:Chain{K}}, trf, rest...) where {K}
    codes = [
        codegen(subtrf, :($trf.chain[$i]), rest...)
        for (i, subtrf) in enumerate(K.parameters)
    ]
    quote
        $(codes...)
    end
end


"""
    Updim{Ins, To}(value)

Create a transformation increasing the dimension by one, by inserting an element `value` at index
`Ins` in each input vector. The final result should have dimension `To`.
"""
struct Updim{Ins, From, To, R<:Real} <: Transform{From, To, R}
    data :: R

    @inline function Updim{Ins, To, R}(data::R) where {Ins, To, R}
        @assert 1 <= Ins <= To
        @assert 1 <= To <= 3
        new{Ins, To-1, To, R}(data)
    end
end

@inline Updim{Ins, To}(data) where {Ins, To} = Updim{Ins, To, typeof(data)}(data)

function codegen(tp::Type{Updim{Ins, From, To, R}}, trf, point) where {Ins, From, To, R}
    dst = reverse(Ins+1 : To)
    src = reverse(Ins : To-1)
    shifts = [:($point[$d] = $point[$s]) for (d,s) in zip(dst, src)]
    quote
        $(shifts...)
        $point[$Ins] = $trf.data
    end
end

function codegen(tp::Type{Updim{Ins, From, To, R}}, trf, point, grad) where {Ins, From, To, R}
    src_cols = [[:($grad[$i,$j]) for i in 1:To] for j in 1:From]

    if To == 1
        new_col = [:(one($R))]
    elseif To == 2
        ((a, b),) = src_cols
        new_col = [b, :(-$a)]
    elseif To == 3
        ((a, c, e), (b, d, f)) = src_cols
        new_col = [:($c*$f - $e*$d), :($e*$b - $a*$f), :($a*$d - $c*$b)]
    end

    dst = Ins+1 : To
    src = Ins : To-1

    quote
        $grad[@SVector(Int[$(dst...)]), :] .= $grad[@SVector(Int[$(src...)]), :]
        $grad[$Ins, :] = zero($R)
        $grad[:, $To] .= @SVector([$(new_col...)])
        $(codegen(tp, trf, point))
    end
end


"""
    Shift(x::SVector)

Create a shifting transformation that adds `x` to each input vector.
"""
struct Shift{D,R} <: Transform{D,D,R}
    data :: SVector{D,R}
end

codegen(tp::Type{<:Shift}, trf, point, rest...) = :($point .+= $trf.data)


"""
    apply!(trf::Transform, x::MVector, [J::MMatrix])

Apply the transform `trf` to the vector `x` and the derivative `J` in-place.
"""
@generated function apply!(trf::Transform{From,To,R}, point::MVector{To,R}) where {From,To,R}
    codegen(trf, :trf, :point)
end

@generated function apply!(trf::Transform{From,To,R}, point::MVector{To,R}, grad::MMatrix{To,To,R}) where {From,To,R}
    codegen(trf, :trf, :point, :grad)
end

end
