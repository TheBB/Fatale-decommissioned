@testset "LocalPoint" begin
    Random.seed!(201812131215)
    func = compile(localpoint(2))

    element = FullElement(Shift(@SVector rand(2)))
    quadpt = @SVector rand(2)
    @test func(element, quadpt) == quadpt

    sub = SubElement(Updim{1,2}(5.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt) == [5.0, quadpt...]

    sub = SubElement(Updim{2,2}(5.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt) == [quadpt..., 5.0]
end


@testset "GlobalPoint" begin
    Random.seed!(201812131219)
    func = compile(globalpoint(2))
    shift = @SVector rand(2)

    element = FullElement(Shift(shift))
    quadpt = @SVector rand(2)
    @test func(element, quadpt) ≈ quadpt + shift

    sub = SubElement(Updim{1,2}(4.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt) ≈ [4.0, quadpt[1]] + shift

    sub = SubElement(Updim{2,2}(4.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt) ≈ [quadpt[1], 4.0] + shift
end


@testset "Monomials" begin
    func = compile(Monomials(localpoint(3), 4))
    element = FullElement(Empty{3,Float64}())
    quadpt = @SVector [1.0, 2.0, 3.0]

    @test func(element, quadpt) ≈ [
        1.0 1.0 1.0  1.0  1.0;
        1.0 2.0 4.0  8.0 16.0;
        1.0 3.0 9.0 27.0 81.0;
    ]
end
