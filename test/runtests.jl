using Random
using StaticArrays
using Test

using Fatale.Transforms


@testset "DimensionTransform" begin
    Random.seed!(201812081344)

    # Dimension 2 -> 4
    data = SVector{4}(rand(Float64, 4))
    trf = DimensionTransform(data, SVector(1,4))
    res = MVector{4,Float64}(undef)

    apply!(res, trf, SVector(1.0, 3.0))
    @test res == [1.0, data[2], data[3], 3.0]

    # Dimension 0 -> 2
    data = SVector{2}(rand(Float64, 2))
    trf = DimensionTransform(data, SVector{0,Int}())
    res = MVector{2,Float64}(undef)

    apply!(res, trf, SVector{0,Float64}())
    @test res == data

    # Dimension 2 -> 2
    data = SVector{2}(rand(Float64, 2))
    trf = DimensionTransform(data, SVector(1,2))
    res = MVector{2,Float64}(undef)

    apply!(res, trf, SVector(1.0, 3.0))
    @test res == [1.0, 3.0]

    # Bounds check
    @test_throws BoundsError DimensionTransform(SVector(1.0, 2.0), SVector(1, 3))
    @test_throws DimensionMismatch DimensionTransform(SVector(1.0, 2.0), SVector(1, 1, 1))
end
