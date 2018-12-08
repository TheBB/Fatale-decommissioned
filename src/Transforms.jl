module Transforms

using StaticArrays

export DimensionTransform
export apply!

struct DimensionTransform{M,N,R<:Real}
    data :: SVector{M,R}
    variates :: SVector{N,Int}

    @inline function DimensionTransform(data::SVector{M,R}, variates::SVector{N,Int}) where {M,N,R<:Real}
        @boundscheck N > M && throw(DimensionMismatch("Index vector too large"))
        @boundscheck checkbounds(data, variates)
        new{M,N,R}(data, variates)
    end
end

function apply!(out::MVector{M,R}, trf::DimensionTransform{M,N,R}, point::StaticVector{N,R}) where {M,N,R<:Real}
    out[:] = trf.data
    out[trf.variates] = point
end

end
