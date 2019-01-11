module Evaluables

import Base: @_inline_meta, Broadcast
import Base.Broadcast: broadcast_shape
import Base.Iterators: flatten, product

using DataStructures
using LinearAlgebra
using StaticArrays
using UnsafeArrays

using ..Transforms
using ..Elements

export Constant, Contract, Elementwise, Inflate, Monomials
export localpoint, localgrad, globalpoint, globalgrad
export compile, storage


"""
    Evaluable{T}

Abstract type representing any function that, when evaluated, produces a value
of type T.
"""
abstract type Evaluable{T} end

restype(::Evaluable{T}) where {T} = T
arguments(::Evaluable) = Evaluable[]

# The default hash/equals behaviour on evaluables is based strictly on
# the types and the arguments involved. This should be sufficient for
# the vast majority of evaluables, but some (e.g. constants) may
# override it.
Base.hash(self::Evaluable, x::UInt64) = hash(typeof(self), hash(arguments(self), x))
Base.:(==)(l::Evaluable, r::Evaluable) = typeof(l) == typeof(r) && arguments(l) == arguments(r)

# Code generation for most evaluables involves simply calling the
# evaluable object directly.
codegen(::Type{<:Evaluable}, self, args...) = :($self($(args...)))


include("evaluables/definitions.jl")
include("evaluables/compilation.jl")
include("evaluables/utility.jl")


end
