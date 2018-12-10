module Transforms

using LinearAlgebra
using StaticArrays

export Transform
export Chain, Updim, Shift
export apply, applygrad


abstract type Transform{From, To, R<:Real} end

@inline fromdims(::Transform{From, To, R}) where {From, To, R} = From
@inline todims(::Transform{From, To, R}) where {From, To, R} = To
@inline eltype(::Transform{From, To, R}) where {From, To, R} = R


struct Chain{K<:Tuple{Vararg{Transform}}, From, To, R<:Real} <: Transform{From, To, R}
    chain :: K

    @inline function Chain{K, From, To, R}(chain::K) where {K<:Tuple{Vararg{Transform}}, From, To, R}
        @assert From == fromdims(chain[1])
        @assert To == todims(chain[end])
        @assert all(R == eltype(link) for link in chain)
        new{K, From, To, R}(chain)
    end
end

@inline function Chain(trfs...)
    Chain{typeof(trfs), fromdims(trfs[1]), todims(trfs[end]), eltype(trfs[1])}(trfs)
end

function codegen(::Type{Chain{K, From, To, R}}, trf, point, grad) where {K, From, To, R}
    ptcodes, gradcodes = Expr[], Expr[]

    for (i, subtrf) in enumerate(K.parameters)
        (ptcode, gradcode) = codegen(subtrf, :($trf.chain[$i]), point, grad)
        point = gensym("point")
        grad = gensym("grad")
        push!(ptcodes, :($point = $ptcode))
        push!(gradcodes, :($grad = $gradcode))
    end

    (:($(ptcodes...); $point), :($(gradcodes...); $grad))
end


struct Updim{Ins, From, To, R<:Real} <: Transform{From, To, R}
    data :: R

    @inline function Updim{Ins, To, R}(data::R) where {Ins, To, R}
        @assert 1 <= Ins <= To
        @assert 1 <= To <= 3
        new{Ins, To-1, To, R}(data)
    end
end

@inline Updim{Ins, To}(data) where {Ins, To} = Updim{Ins, To, typeof(data)}(data)

function codegen(tp::Type{Updim{Ins, From, To, R}}, trf, point, grad) where {Ins, From, To, R}
    exprs = Expr[:($point[$i]) for i in 1:From]
    insert!(exprs, Ins, :($trf.data))

    cols = [[:($grad[$i,$j]) for i in 1:From] for j in 1:From]
    for col in cols
        insert!(col, Ins, :(zero($R)))
    end

    if To == 1
        push!(cols, [:(one($R))])
    elseif To == 2
        ((a, b),) = cols
        push!(cols, [b, :(-$a)])
    elseif To == 3
        ((a, c, e), (b, d, f)) = cols
        push!(cols, [:($c*$f - $e*$d), :($e*$b - $a*$f), :($a*$d - $c*$b)])
    end

    ptcode = :(SVector{$To,$R}($(exprs...)))
    gradcode = :(SMatrix{$To,$To,$R}($(Iterators.flatten(cols)...)))
    (ptcode, gradcode)
end


struct Shift{D,R} <: Transform{D,D,R}
    data :: SVector{D,R}
end

codegen(tp::Type{<:Shift}, trf, point, grad) = (:($point + $trf.data), grad)


@generated function apply(trf::Transform{From,To,R}, point::SVector{From,R}) where {From,To,R}
    (ptcode, _) = codegen(trf, :trf, :point, :grad)
    :(return $ptcode :: SVector{$To,$R})
end

@generated function applygrad(trf::Transform{From,To,R}, point::SVector{From,R}) where {From,To,R}
    (ptcode, gradcode) = codegen(trf, :trf, :point, :grad)
    quote
        grad = SMatrix{$From,$From,$R}(I)
        return ($ptcode, $gradcode) :: Tuple{SVector{$To,$R}, SMatrix{$To,$To,$R}}
    end
end


end
