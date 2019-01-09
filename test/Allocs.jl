macro noallocs(expr)
    quote
        $(esc(expr))
        trial = $(esc(expr))
        @test trial.allocs == 0
    end
end


function _trfchain()
    trf = Chain(Updim{1,2}(1.0), Shift(SVector(4.0, 5.0)))
    initial = @SVector rand(1)
    igrad = SMatrix{1,1,Float64}(I)
    @benchmark $trf($initial, $igrad)
end

@testset "Transform chain" begin
    @noallocs _trfchain()
end


function _globalpoint()
    func = compile(globalpoint(2))
    element = FullElement(Shift(@SVector rand(2)))
    quadpt = @SVector rand(2)
    @benchmark $func($element, $quadpt)
end

@testset "GlobalPoint" begin
    @noallocs _globalpoint()
end


function _monomials()
    func = compile(Monomials(localpoint(3), 4))
    element = FullElement(Empty{3,Float64}())
    quadpt = @SVector [1.0, 2.0, 3.0]
    @benchmark $func($element, $quadpt)
end

@testset "Monomials" begin
    @noallocs _monomials()
end