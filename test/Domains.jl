@testset "TensorDomain" begin
    domain = TensorDomain(2, 3, 4)
    @test size(domain) == (2, 3, 4)

    elt = domain[1, 2, 3]
    @test isa(elt, Element{3})
    @test isa(elt, FullElement{3, Shift{3,Float64}})
    @test elt.transform.data == [0.0, 1.0, 2.0]
end
