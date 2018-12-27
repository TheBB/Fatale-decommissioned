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

include("evaluables/types.jl")
include("evaluables/definitions.jl")
include("evaluables/compilation.jl")
include("evaluables/utility.jl")

end
