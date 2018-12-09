module Transforms

using StaticArrays

export Updim, Shift
export apply!


struct Updim{M,N,R<:Real}
    data :: SVector{M,R}
    variates :: SVector{N,Int}

    @inline function Updim(data::SVector{M,R}, variates::SVector{N,Int}) where {M,N,R<:Real}
        @boundscheck N > M && throw(DimensionMismatch("Index vector too large"))
        @boundscheck checkbounds(data, variates)
        new{M,N,R}(data, variates)
    end
end

function apply!(out::MVector{M,R}, trf::Updim{M,N,R}, point::StaticVector{N,R}) where {M,N,R<:Real}
    out[:] = trf.data
    out[trf.variates] = point
end


struct Shift{M,R<:Real}
    data :: SVector{M,R}
end

function apply!(out::MVector{M,R}, trf::Shift{M,R}, point::StaticVector{M,R}) where {M,R<:Real}
    out[:] = point + trf.data
end

end
