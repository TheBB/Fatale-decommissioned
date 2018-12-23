@testset "LocalCoords" begin
    Random.seed!(201812131215)
    func = compile(LocalCoords(2))

    element = FullElement(Shift(@SVector rand(2)))
    quadpt = @SVector rand(2)
    @test func(element, quadpt).point == quadpt

    sub = SubElement(Updim{1,2}(5.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt).point == [5.0, quadpt[1]]

    sub = SubElement(Updim{2,2}(5.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt).point == [quadpt[1], 5.0]
end


@testset "GlobalCoords" begin
    Random.seed!(201812131219)
    func = compile(GlobalCoords(2))
    shift = @SVector rand(2)

    element = FullElement(Shift(shift))
    quadpt = @SVector rand(2)
    @test func(element, quadpt).point ≈ quadpt + shift

    sub = SubElement(Updim{1,2}(4.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt).point ≈ [4.0, quadpt[1]] + shift

    sub = SubElement(Updim{2,2}(4.0), element)
    quadpt = @SVector rand(1)
    @test func(sub, quadpt).point ≈ [quadpt[1], 4.0] + shift
end
