using BenchmarkTools
using StaticArrays

using Fatale.Domains
using Fatale.Evaluables


dom = TensorDomain(1, 1)


bs = basis(dom, Lagrange, 2)
xs = bs[1,:]
ys = reshape(bs[2,:], 1, :)

cbs = compile(ys)

element = dom[1, 1]
quadpt = @SVector [0.7, 0.6]

@show cbs(element, quadpt)
@btime $cbs($element, $quadpt)
