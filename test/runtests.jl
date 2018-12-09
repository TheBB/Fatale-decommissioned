using Random
using StaticArrays
using Test

using Fatale.Transforms


@testset "Updim" begin
    Random.seed!(201812081344)

    # Dimension 2 -> 3
    initial = rand(Float64, 3)

    trf = Updim{3,1,Float64}(rand(Float64))
    res = MVector{3}(initial)
    apply!(res, trf)
    @test res == [trf.data, initial[1], initial[2]]

    trf = Updim{3,2,Float64}(rand(Float64))
    res = MVector{3}(initial)
    apply!(res, trf)
    @test res == [initial[1], trf.data, initial[2]]

    trf = Updim{3,3,Float64}(rand(Float64))
    res = MVector{3}(initial)
    apply!(res, trf)
    @test res == [initial[1], initial[2], trf.data]

    # Dimension 0 -> 1
    trf = Updim{1,1,Float64}(rand(Float64))
    res = MVector{1,Float64}(undef)
    apply!(res, trf)
    @test res == [trf.data]
end


@testset "Shift" begin
    Random.seed!(201812091010)

    trf = Shift(SVector{4}(rand(Float64, 4)))
    initial = rand(Float64, 4)
    res = MVector{4}(initial)

    apply!(res, trf)
    @test res == initial + trf.data
end
