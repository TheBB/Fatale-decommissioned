Base.eltype(::Type{<:Evaluable{T}}) where {T} = eltype(T)
Base.ndims(::Type{<:Evaluable{T}}) where {T} = ndims(T)
Base.size(::Type{<:Evaluable{T}}) where {T} = size(T)
Base.size(::Type{<:Evaluable{T}}, i) where {T} = size(T, i)

Base.eltype(::Evaluable{T}) where {T} = eltype(T)
Base.ndims(::Evaluable{T}) where {T} = ndims(T)
Base.size(::Evaluable{T}) where {T} = size(T)
Base.size(::Evaluable{T}, i) where {T} = size(T, i)

marray(size, eltype) = MArray{Tuple{size...}, eltype, length(size), prod(size)}
sarray(size, eltype) = SArray{Tuple{size...}, eltype, length(size), prod(size)}


function Base.getproperty(self::Evaluable{T}, v::Symbol) where {T<:NamedTuple}
    index = findfirst(x->x==v, T.parameters[1])
    index == nothing && return getfield(self, v)
    GetProperty{T.parameters[2].parameters[index], v}(self)
end

Base.getindex(self::Evaluable, inds...) = GetIndex(self, inds...)


localpoint(n) = LocalCoords(n).point
localgrad(n) = LocalCoords(n).grad
globalpoint(n) = GlobalCoords(n).point
globalgrad(n) = GlobalCoords(n).grad
