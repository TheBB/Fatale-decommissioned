module Evaluables

using StaticArrays
using ..Elements

export LocalCoords, GlobalCoords
export compile

include("evaluables/types.jl")
include("evaluables/definitions.jl")
include("evaluables/compilation.jl")

end
