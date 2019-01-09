module Evaluables

import Base: @_inline_meta
import Base.Iterators: flatten, product

using DataStructures
using LinearAlgebra
using StaticArrays

using ..Transforms
using ..Elements

export Constant, Contract, Monomials
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


include("evaluables/definitions.jl")
include("evaluables/compilation.jl")
include("evaluables/utility.jl")


end
