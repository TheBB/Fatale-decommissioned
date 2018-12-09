module Transforms

using StaticArrays

export Transform
export Updim, Shift
export apply!


abstract type Transform{M,R<:Real} end


struct Updim{M,N,R} <: Transform{M,R}
    data :: R
end

function codegen(::Type{Updim{M,N,R}}, trf, point) where {M,N,R}
    exprs = [:($point[$(i)] = $point[$(i-1)]) for i in M:-1:N+1]
    quote
        $(exprs...)
        $point[$N] = $trf.data
    end
end


struct Shift{M,R<:Real} <: Transform{M,R}
    data :: SVector{M,R}
end

function codegen(::Type{Shift{M,R}}, trf, point) where {M,R}
    :($point .+= trf.data)
end


@generated function apply!(point::MVector{M,R}, trf::Transform{M,R}) where {M,R}
    codegen(trf, :trf, :point)
end

end
