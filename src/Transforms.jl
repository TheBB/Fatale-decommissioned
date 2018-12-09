module Transforms

using StaticArrays

export Transform
export Updim, Shift
export apply!


abstract type Transform{M,R<:Real} end


struct Updim{M,N,R} <: Transform{M,R}
    data :: R
end

function codegen(::Type{<:Updim{M,N}}, trf, point) where {M,N}
    exprs = [:($point[$i] = $point[$(i-1)]) for i in M:-1:N+1]
    quote
        $(exprs...)
        $point[$N] = $trf.data
    end
end

codegen(tp::Type{<:Updim{1,1}}, trf, point, grad) = codegen(tp, trf, point)

function codegen(tp::Type{<:Updim{2,N}}, trf, point, grad) where {N}
    exprs = [:($grad[$i,:] .= $grad[$(i-1),:]) for i in 2:-1:N+1]
    quote
        $(codegen(tp, trf, point))
        $(exprs...)
        $grad[$N, :] .= 0.0
        $grad[1,2] = $grad[2,1]
        $grad[2,2] = -$grad[1,1]
    end
end


struct Shift{M,R<:Real} <: Transform{M,R}
    data :: SVector{M,R}
end

function codegen(::Type{<:Shift}, trf, point)
    :($point .+= trf.data)
end

codegen(tp::Type{<:Shift}, trf, point, grad) = codegen(tp, trf, point)


@generated function apply!(point::MVector{M,R}, trf::Transform{M,R}) where {M,R}
    codegen(trf, :trf, :point)
end

@generated function apply!(point::MVector{M,R}, grad::MMatrix{M,M,R}, trf::Transform{M,R}) where {M,R}
    codegen(trf, :trf, :point, :grad)
end

end
