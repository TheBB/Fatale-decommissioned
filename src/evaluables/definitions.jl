"""
    LocalCoords{N, T} <: VectorEvaluable{N, T}

Function returning the local (reference) coordinates of the quadrature point.
"""
struct LocalCoords{N,T} <: VectorEvaluable{N,T} end
storage(::LocalCoords{N,T}) where {N,T} = (point = :(MVector{$N,$T}(undef)),)

function codegen(::LocalCoords; point)
    quote
        $point[1:length(quadpt)] .= quadpt
        apply!(dimtrans(element), $point)
        $point
    end
end


"""
    GlobalCoords{N, T} <: VectorEvaluable{N, T}

Function returning the global (physical) coordinates of the quadrature point.
"""
struct GlobalCoords{N,T} <: VectorEvaluable{N,T} end
storage(::GlobalCoords{N,T}) where {N,T} = (point = :(MVector{$N,$T}(undef)),)

function codegen(::GlobalCoords; point)
    quote
        $point[1:length(quadpt)] .= quadpt
        apply!(globtrans(element), $point)
        $point
    end
end
