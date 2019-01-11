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


Broadcast.broadcasted(::typeof(*), l::Evaluable) = l
Broadcast.broadcasted(::typeof(*), l::Product, r::Evaluable) = Product(arguments(l)..., r)
Broadcast.broadcasted(::typeof(*), l::Evaluable, r::Product) = Product(l, arguments(r)...)
Broadcast.broadcasted(::typeof(*), l::Evaluable, r::Evaluable) = Product(l, r)

function Base.getproperty(self::Evaluable{T}, v::Symbol) where {T<:NamedTuple}
    index = findfirst(x->x==v, T.parameters[1])
    index == nothing && return getfield(self, v)
    GetProperty{T.parameters[2].parameters[index], v}(self)
end

function Base.getindex(self::Evaluable, inds...)
    @assert length(inds) == ndims(restype(self))
    @assert all(
        isa(i, Colon) || isa(i, Integer) && 1 <= i <= s
        for (i, s) in zip(inds, size(self))
    )

    arraytype(self) == MArray && return UnsafeGetIndex(self, inds...)
    error("getindex not defined for this array type")
end

function Base.reshape(self::Evaluable, size...)
    size = collect(Union{Int,Colon}, size)
    colon = findfirst(x->x==:, size)
    if colon != nothing && length(size) > 1
        size[colon] = div(length(self), prod(k for k in size if !isa(k, Colon)))
    elseif colon != nothing
        size[colon] = length(self)
    end

    arraytype(self) == MArray && return UnsafeReshape(self, size...)
    error("reshape not defined for this array type")
end


localpoint(n) = LocalCoords(n).point
localgrad(n) = LocalCoords(n).grad
globalpoint(n) = GlobalCoords(n).point
globalgrad(n) = GlobalCoords(n).grad
