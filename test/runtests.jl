using LinearAlgebra
using Random
using StaticArrays
using Test

using Fatale.Transforms
using Fatale.Elements


@testset "Transforms" begin
    include("Transforms.jl")
end

@testset "Elements" begin
    include("Elements.jl")
end
