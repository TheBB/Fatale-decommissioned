import UnsafeArrays
using StaticArrays


# A couple of hacks needed to make unsafe views into mutable static
# arrays work.
UnsafeArrays._require_one_based_indexing(::MArray) = nothing
Base.@propagate_inbounds UnsafeArrays.unsafe_uview(a::MArray{N,T}) where {N,T} =
    UnsafeArrays._maybe_unsafe_uview(Val{isbitstype(T)}(), a)
