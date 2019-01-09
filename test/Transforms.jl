@testset "Updim" begin
    Random.seed!(201812081344)

    # Dimension 2 -> 3
    initial = @SVector rand(2)

    trf = Updim{1,3}(rand(Float64))
    @test trf(initial) == [trf.data, initial...]

    trf = Updim{2,3}(rand(Float64))
    @test trf(initial) == [initial[1], trf.data, initial[2]]

    trf = Updim{3,3}(rand(Float64))
    @test trf(initial) == [initial..., trf.data]

    # Dimension 1 -> 2
    initial = @SVector rand(1)
    igrad = SMatrix{1,1}(1.0)

    trf = Updim{1,2}(rand(Float64))
    (point, grad) = trf(initial, igrad)
    @test point == [trf.data, initial...]
    @test grad == [0.0 1.0; 1.0 0.0]

    trf = Updim{2,2}(rand(Float64))
    (point, grad) = trf(initial, igrad)
    @test point == [initial..., trf.data]
    @test grad == [1.0 0.0; 0.0 -1.0]

    # Dimension 0 -> 1
    trf = Updim{1,1}(rand(Float64))
    point = SVector{0,Float64}()
    @test trf(point) == [trf.data]
end


@testset "Shift" begin
    Random.seed!(201812091010)

    trf = Shift(@SVector rand(4))
    initial = @SVector rand(4)
    @test trf(initial) == initial + trf.data

    grad = SMatrix{4,4,Float64}(I)
    (point, grad) = trf(initial, grad)
    @test point == initial + trf.data
    @test grad == I
end


@testset "Chain" begin
    Random.seed!(201812091503)
    trf = Chain(Updim{1,2}(1.0), Shift(SVector(4.0, 5.0)))

    initial = @SVector rand(1)
    @test trf(initial) == [5.0, initial[1] + 5.0]

    igrad = SMatrix{1,1,Float64}(I)
    (point, grad) = trf(initial, igrad)
    @test point == [5.0, initial[1] + 5.0]
    @test grad == [0.0 1.0; 1.0 0.0]
end
