Base.eltype(::Type{<:ArrayEvaluable{S, T}}) where {S, T} = T
Base.ndims(::Type{<:ArrayEvaluable{S, T, N}}) where {S, T, N} = N
Base.size(::Type{<:ArrayEvaluable{S}}) where {S} = Tuple(S.parameters)
Base.size(::Type{<:ArrayEvaluable{S}}, i) where {S} = S.parameters[i]

Base.eltype(::ArrayEvaluable{S, T}) where {S, T} = T
Base.ndims(::ArrayEvaluable{S, T, N}) where {S, T, N} = N
Base.size(::ArrayEvaluable{S}) where {S} = Tuple(S.parameters)
Base.size(::ArrayEvaluable{S}, i) where {S} = S.parameters[i]


function Base.getproperty(self::Evaluable{T}, v::Symbol) where {T<:NamedTuple}
    index = findfirst(x->x==v, T.parameters[1])
    index == nothing && return getfield(self, v)
    GetProperty{T.parameters[2].parameters[index], v}(self)
end


localpoint(n) = LocalCoords(n).point
localgrad(n) = LocalCoords(n).grad
globalpoint(n) = GlobalCoords(n).point
globalgrad(n) = GlobalCoords(n).grad
