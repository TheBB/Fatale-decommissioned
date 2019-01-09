macro noallocs(expr)
    quote
        $(esc(expr))
        trial = $(esc(expr))
        @test trial.allocs == 0
    end
end


function trfchain()
    trf = Chain(Updim{1,2}(1.0), Shift(SVector(4.0, 5.0)))
    initial = @SVector rand(1)
    igrad = SMatrix{1,1,Float64}(I)
    @benchmark $trf($initial, $igrad)
end

@testset "Transform chain" begin
    @noallocs trfchain()
end
