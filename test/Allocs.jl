macro noallocs(expr)
    quote
        $(esc(expr))
        trial = $(esc(expr))
        @test trial.allocs == 0
    end
end

macro bench(expr)
    :(@benchmark $expr samples=1 evals=1)
end


function _trfchain()
    trf = Chain(Updim{1,2}(1.0), Shift(SVector(4.0, 5.0)))
    initial = @SVector rand(1)
    igrad = SMatrix{1,1,Float64}(I)
    @bench $trf($initial, $igrad)
end

@testset "Transform chain" begin
    @noallocs _trfchain()
end


function _tensordofmap()
    domain = TensorDomain(2, 3, 4)
    dofmap = TensorDofMap(size(domain), (1, 2, 1), (3, 3, 2))
    @bench $dofmap[1, 3, 2]
end

@testset "TensorDofMap" begin
    @noallocs _tensordofmap()
end


function _globalpoint()
    func = optimize(globalpoint(2))
    element = FullElement(Shift(@SVector rand(2)))
    quadpt = @SVector rand(2)
    @bench $func($element, $quadpt)
end

@testset "GlobalPoint" begin
    @noallocs _globalpoint()
end


function _monomials()
    func = optimize(Monomials(localpoint(3), 4))
    element = FullElement(Empty{3,Float64}())
    quadpt = @SVector [1.0, 2.0, 3.0]
    @bench $func($element, $quadpt)
end

@testset "Monomials" begin
    @noallocs _monomials()
end


function _lagbasis()
    domain = TensorDomain(1, 1)
    func = optimize(localbasis(domain, Lagrange, 3))
    element = domain[1, 1]
    quadpt = @SVector [0.3, 0.7]
    @bench $func($element, $quadpt)
end

@testset "Lagrangian Basis" begin
    @noallocs _lagbasis()
end


function _blockeval()
    domain = TensorDomain(1)
    func = compile(globalbasis(domain, Lagrange, 2))
    element = domain[1]
    quadpt = @SVector [0.4]
    @bench begin
        indices($func[1], $element)
        $func[1]($element, $quadpt)
    end
end

@testset "Block Evaluation" begin
    @noallocs _blockeval()
end
