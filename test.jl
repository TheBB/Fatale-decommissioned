using BenchmarkTools
using StaticArrays

using Fatale.Domains
using Fatale.Evaluables


dom = TensorDomain(1, 1)


bs = basis(dom, Lagrange, 2)
cbs = compile(bs)

element = dom[1, 1]
quadpt = @SVector [0.6, 0.6]

@show cbs(element, quadpt)
@btime $cbs($element, $quadpt)
