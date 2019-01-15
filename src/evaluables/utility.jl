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


_repr(::Type{T}) where {T<:StaticArray} = string("(", join(size(T), ","), ")")

function _repr(::Type{T}) where {T<:Tuple}
    list = join((_repr(param) for param in T.parameters), ", ")
    string("(", list, ")")
end

function _repr(::Type{T}) where {T<:NamedTuple}
    names = T.parameters[1]
    values = T.parameters[2].parameters
    entries = collect(string(name, "=", _repr(value)) for (name, value) in zip(names, values))
    list = join(entries, ", ")
    string("{", list, "}")
end

function Base.show(io::IO, self::Evaluable{T}) where T
    print(io, string(typeof(self).name.name), _repr(T))
end

Base.show(io::IO, self::CompiledBlock) = print(io, "Blk(", self.indices, ", ", self.data, ")")
Base.show(io::IO, self::CompiledBlocks) = print(io, "CBlks(", self.blocks..., ")")


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


# =============================================================================
# Reshaping

function _reshape(self::Evaluable, shape::NTuple{N, Int}) where N
    size(self) == shape && return self
    arraytype(self) == MArray && return UnsafeReshape(self, shape...)
    error("reshape not defined for this array type")
end

_reshape(self::Reshape, shape::NTuple{N, Int}) where N = _reshape(arguments(self)[1], shape)

Base.reshape(self::Evaluable, shape::Vararg{Int}) = _reshape(self, shape)

# Allow a single colon
function Base.reshape(self::Evaluable, shape::Vararg{Union{Int,Colon}})
    shape = collect(Union{Int,Colon}, shape)
    colon = findfirst(x -> x == :, shape)
    if length(shape) > 1
        shape[colon] = div(length(self), prod(k for k in shape if !isa(k, Colon)))
    else
        shape[colon] = length(self)
    end
    @assert all(s isa Int for s in shape)
    _reshape(self, (shape...,))
end

const .. = Val{:..}()

# Allow a single ..
function Base.reshape(self::Evaluable, shape::Vararg{Union{Int,Val{:..}}})
    shape = collect(Union{Int,Colon}, shape)
    ellipsis = findfirst(x -> x isa Val{:..}, shape)
    shape = vcat(shape[1:ellipsis-1], collect(size(self)), shape[ellipsis+1:end])
    @assert all(s isa Int for s in shape)
    _reshape(self, (shape...,))
end


# =============================================================================
# Convenience constructors

localpoint(n) = LocalCoords(n).point
localgrad(n) = LocalCoords(n).grad
globalpoint(n) = GlobalCoords(n).point
globalgrad(n) = GlobalCoords(n).grad
