@testset "Simplex" begin
    @test ndims(Simplex{1}) == 1

    line = Simplex{1}()
    @test ndims(line) == 1

    (pts, wts) = quadrule(line, 3)
    @test typeof(pts) == Vector{SVector{1,Float64}}
    @test typeof(wts) == Vector{Float64}
    @test pts ≈ [[0.1127016653792583], [0.5], [0.8872983346207417]]
    @test wts ≈ [0.2777777777777777, 0.4444444444444444, 0.2777777777777777]
end


@testset "Tensor" begin
    line = Simplex{1}()
    square = Tensor(line, line)

    @test ndims(square) == 2
    @test ndims(typeof(square)) == 2

    (pts, wts) = quadrule(square, 2)
    @test typeof(pts) == Vector{SVector{2,Float64}}
    @test pts ≈ [
        [0.21132486540518708, 0.21132486540518708],
        [0.78867513459481290, 0.21132486540518708],
        [0.21132486540518708, 0.78867513459481290],
        [0.78867513459481290, 0.78867513459481290],
    ]
    @test wts ≈ [0.25, 0.25, 0.25, 0.25]
end
