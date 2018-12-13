"""
    Evaluable{T}

Abstract type representing any function that, when evaluated, produces a value
of type T.
"""
abstract type Evaluable{T} end


"""
    ArrayEvaluable{S, T, N, L} <: Evaluable{SArray{S, T, N, L}}

Abstract type representing any function that, when evaluated, produces a static
array value. The type parameters correspond to those of the SArray type: size,
element type, number of dimensions and total length.
"""
const ArrayEvaluable{S, T, N, L} = Evaluable{SArray{S, T, N, L}}


"""
    ScalarEvaluable{T} <: ArrayEvaluable

Abstract type representing any function that, when evaluated, returns a scalar
of type T.
"""
const ScalarEvaluable{T} = ArrayEvaluable{Tuple{}, T, 0, 0}


"""
    VectorEvaluable{N, T} <: ArrayEvaluable

Abstract type representing any function that, when evaluated, returns a vector
of length N with elements of type T.
"""
const VectorEvaluable{N, T} = ArrayEvaluable{Tuple{N}, T, 1, N}


"""
    MatrixEvaluable{M, N, T} <: ArrayEvaluable

Abstract type representing any function that, when evaluated, returns a matrix
of size M x N with elements of type T.
"""
const MatrixEvaluable{M, N, T, L} = ArrayEvaluable{Tuple{M,N}, T, 2, L}
