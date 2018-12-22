"""
    LocalCoords{N, T}

Function returning the local (reference) coordinates of the quadrature point.
"""
struct LocalCoords{N,T,L} <:
    Evaluable{Tuple{VectorEvaluable{N,T}, SquareMatrixEvaluable{N,T,L}}}
    LocalCoords{N,T}() where {N,T} = new{N,T,N*N}()
end

_storage(::LocalCoords{N,T}) where {N,T} = (MVector{N,T}(undef), MMatrix{N,N,T}(undef))

function evaluate(::LocalCoords{N}, element, quadpt::SVector{M}, storage) where {M,N}
    storage.mine[1][1:M] .= quadpt
    storage.mine[2] .= Matrix{Float64}(I, N, N)
    apply!(dimtrans(element), storage.mine[1], storage.mine[2])
    storage.mine
end


"""
    GlobalCoords{N, T}

Function returning the global (physical) coordinates of the quadrature point.
"""
struct GlobalCoords{N,T,L} <:
    Evaluable{Tuple{VectorEvaluable{N,T}, SquareMatrixEvaluable{N,T,L}}}
    GlobalCoords{N,T}() where {N,T} = new{N,T,N*N}()
end

_storage(::GlobalCoords{N,T}) where {N,T} = (MVector{N,T}(undef), MMatrix{N,N,T}(undef))

function evaluate(::GlobalCoords{N}, element, quadpt::SVector{M}, storage) where {M,N}
    storage.mine[1][1:M] .= quadpt
    storage.mine[2] .= Matrix{Float64}(I, N, N)
    apply!(globtrans(element), storage.mine[1], storage.mine[2])
    storage.mine
end
