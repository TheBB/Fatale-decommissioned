using BenchmarkTools
using LinearAlgebra
using Random
using StaticArrays
using Test

using Fatale.Transforms
using Fatale.Elements
using Fatale.Domains
using Fatale.Evaluables


@testset "Transforms" begin
    include("Transforms.jl")
end

@testset "Elements" begin
    include("Elements.jl")
end

@testset "Domain" begin
    include("Domains.jl")
end

@testset "Evaluables" begin
    include("Evaluables.jl")
end

@testset "Allocs" begin
    include("Allocs.jl")
end
