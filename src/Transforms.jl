module Transforms

using StaticArrays

export Updim, Shift
export apply!


struct Updim{M,N,R<:Real}
    data :: R
end

function apply!(point::MVector{M,R}, trf::Updim{M,N,R}) where {M,N,R<:Real}
    point[N+1:M] = point[N:M-1]
    point[N] = trf.data
end


struct Shift{M,R<:Real}
    data :: SVector{M,R}
end

function apply!(point::MVector{M,R}, trf::Shift{M,R}) where {M,R<:Real}
    point .+= trf.data
end

end
