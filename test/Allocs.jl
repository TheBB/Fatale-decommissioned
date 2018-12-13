macro checkallocs(code)
    quote
        $code
        trial = @benchmark $code samples=1 evals=1
        @test trial.allocs == 0
    end
end


function mktransforms()
    t1 = Updim{1, 1}(0.5)
    t2 = Updim{2, 2}(1.6)
    t3 = Updim{1, 3}(4.4)

    diff = (4, 5, 1)
    t4 = SVector{3,Float64}(diff...) - 1

    chain = Chain(t1, t2, t3, Shift(t4))
end

@testset "Transforms" begin
    @checkallocs mktransforms()
end


function mkelements()
    t1 = Shift(SVector(1.0, 2.0, 3.0))
    full = FullElement(t1)

    t2 = Updim{2,3}(1.0)
    sub = SubElement(t2, full)
end

@testset "Elements" begin
    @checkallocs mkelements()
end


function iterdomain()
    domain = TensorDomain(2, 2, 2)
    indices = eachindex(domain)

    for I in indices
        element = domain[I]
    end
end

@testset "Domains" begin
    @checkallocs iterdomain()
end


@testset "LocalCoords" begin
    func = compile(LocalCoords{2, Float64}())
    element = FullElement(Shift(@SVector rand(2)))
    quadpt = @SVector rand(2)

    # TODO: Why can't I checkallocs this?
    func(element, quadpt)
    trial = @benchmark $func($element, $quadpt) samples=1 evals=1
    @test trial.allocs == 0
end


@testset "GlobalCoords" begin
    func = compile(GlobalCoords{3, Float64}())
    element = SubElement(Updim{1,3}(4.0), FullElement(Shift(@SVector rand(3))))
    quadpt = @SVector rand(2)

    # TODO: Why can't I checkallocs this?
    func(element, quadpt)
    trial = @benchmark $func($element, $quadpt) samples=1 evals=1
    @test trial.allocs == 0
end
