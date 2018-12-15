@testset "Updim" begin
    Random.seed!(201812081344)

    # Dimension 2 -> 3
    initial = rand(2)

    trf = Updim{1,3}(rand(Float64))
    point = MVector{3,Float64}(initial..., 0.0)
    apply!(trf, point)
    @test point == [trf.data, initial[1], initial[2]]

    trf = Updim{2,3}(rand(Float64))
    point = MVector{3,Float64}(initial..., 0.0)
    apply!(trf, point)
    @test point == [initial[1], trf.data, initial[2]]

    trf = Updim{3,3}(rand(Float64))
    point = MVector{3,Float64}(initial..., 0.0)
    apply!(trf, point)
    @test point == [initial[1], initial[2], trf.data]

    # Dimension 1 -> 2
    initial = rand()

    trf = Updim{1,2}(rand(Float64))
    point = MVector{2,Float64}(initial, 0.0)
    grad = MMatrix{2,2,Float64}(I)
    apply!(trf, point, grad)
    @test point == [trf.data, initial]
    @test grad == [0.0 1.0; 1.0 0.0]

    trf = Updim{2,2}(rand(Float64))
    point = MVector{2,Float64}(initial, 0.0)
    grad = MMatrix{2,2,Float64}(I)
    apply!(trf, point, grad)
    @test point == [initial[1], trf.data]
    @test grad == [1.0 0.0; 0.0 -1.0]

    # Dimension 0 -> 1
    trf = Updim{1,1}(rand(Float64))
    point = MVector{1,Float64}(0.0)
    apply!(trf, point)
    @test point == [trf.data]
end


@testset "Shift" begin
    Random.seed!(201812091010)

    trf = Shift(@SVector rand(4))
    initial = @SVector rand(4)
    point = MVector(initial...)
    apply!(trf, point)
    @test point == initial + trf.data

    point = MVector(initial...)
    grad = MMatrix{4,4,Float64}(I)
    apply!(trf, point, grad)
    @test point == initial + trf.data
    @test grad == I
end


@testset "Chain" begin
    Random.seed!(201812091503)
    trf = Chain(Updim{1,2}(1.0), Shift(SVector(4.0, 5.0)))

    initial = rand()
    point = MVector{2,Float64}(initial, 0.0)
    apply!(trf, point)
    @test point == [5.0, initial + 5.0]

    point = MVector{2,Float64}(initial, 0.0)
    grad = MMatrix{2,2,Float64}(I)
    apply!(trf, point, grad)
    @test point == [5.0, initial + 5.0]
    @test grad == [0.0 1.0; 1.0 0.0]
end
