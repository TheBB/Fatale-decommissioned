Base.eltype(::Type{<:Evaluable{T}}) where {T} = eltype(T)
Base.length(::Type{<:Evaluable{T}}) where {T} = length(T)
Base.ndims(::Type{<:Evaluable{T}}) where {T} = ndims(T)
Base.size(::Type{<:Evaluable{T}}) where {T} = size(T)
Base.size(::Type{<:Evaluable{T}}, i) where {T} = size(T, i)

Base.eltype(::Evaluable{T}) where {T} = eltype(T)
Base.length(::Evaluable{T}) where {T} = length(T)
Base.ndims(::Evaluable{T}) where {T} = ndims(T)
Base.size(::Evaluable{T}) where {T} = size(T)
Base.size(::Evaluable{T}, i) where {T} = size(T, i)

arraytype(::Evaluable{T}) where {T} = error("Unknown array type: $T")
arraytype(::Evaluable{<:SArray}) = SArray
arraytype(::Evaluable{<:MArray}) = MArray

array(size, eltype, root) = root{Tuple{size...}, eltype, length(size), prod(size)}
marray(size, eltype) = array(size, eltype, MArray)
sarray(size, eltype) = array(size, eltype, SArray)


function Base.getproperty(self::Evaluable{T}, v::Symbol) where {T<:NamedTuple}
    index = findfirst(x->x==v, T.parameters[1])
    index == nothing && return getfield(self, v)
    GetProperty{T.parameters[2].parameters[index], v}(self)
end

Base.getindex(self::Evaluable, inds...) = GetIndex(self, inds...)

Base.reshape(self::Evaluable, size...) = Reshape(self, size...)


localpoint(n) = LocalCoords(n).point
localgrad(n) = LocalCoords(n).grad
globalpoint(n) = GlobalCoords(n).point
globalgrad(n) = GlobalCoords(n).grad
