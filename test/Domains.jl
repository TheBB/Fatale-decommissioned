@testset "TensorDofMap" begin
    domain = TensorDomain(2, 3, 4)

    # 1: quadratic, full continuity (4 functions)
    # 2: quadratic, lowest continuity (7 functions)
    # 3: linear, full continuity (5 functions)
    dofmap = TensorDofMap(size(domain), (1, 2, 1), (3, 3, 2))

    base = [1, 2, 3, 5, 6, 7, 9, 10, 11, 29, 30, 31, 33, 34, 35, 37, 38, 39]

    @test dofmap[1,1,1] == base
    @test dofmap[2,1,1] == base .+ 1
    @test dofmap[1,2,1] == base .+ 8
    @test dofmap[2,2,1] == base .+ 9
    @test dofmap[1,3,1] == base .+ 16
    @test dofmap[2,3,1] == base .+ 17

    @test dofmap[1,1,2] == base .+ 28
    @test dofmap[2,1,2] == base .+ 29
    @test dofmap[1,2,2] == base .+ 36
    @test dofmap[2,2,2] == base .+ 37
    @test dofmap[1,3,2] == base .+ 44
    @test dofmap[2,3,2] == base .+ 45

    @test dofmap[1,1,3] == base .+ 56
    @test dofmap[2,1,3] == base .+ 57
    @test dofmap[1,2,3] == base .+ 64
    @test dofmap[2,2,3] == base .+ 65
    @test dofmap[1,3,3] == base .+ 72
    @test dofmap[2,3,3] == base .+ 73

    @test dofmap[1,1,4] == base .+ 84
    @test dofmap[2,1,4] == base .+ 85
    @test dofmap[1,2,4] == base .+ 92
    @test dofmap[2,2,4] == base .+ 93
    @test dofmap[1,3,4] == base .+ 100
    @test dofmap[2,3,4] == base .+ 101
end


@testset "TensorDomain" begin
    domain = TensorDomain(2, 3, 4)
    @test size(domain) == (2, 3, 4)

    elt = domain[1, 2, 3]
    @test isa(elt, Element{3})
    @test isa(elt, FullElement{3, Shift{3,Float64}})
    @test elt.transform.data == [0.0, 1.0, 2.0]
end
