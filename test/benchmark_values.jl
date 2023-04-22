#!/usr/bin/julia

using MSSim
using BenchmarkTools

for (cumdis, area_mode, grad) in Iterators.product((false, true), (false, true),
                                                   (false, true))
    @show (cumdis, area_mode, grad)
    @btime MSSim.SymLinear.SegInt.compute_values(
        $(1.0), $(1.0), $(1.0), $(1.0), $(1.0),
        $(Val(cumdis)), $(Val(area_mode)), $(Val(grad)))
    @btime MSSim.SymLinear.SegInt.compute_values(
        $(1.0), $(1.0), $(MSSim.Utils.Zero()), $(1.0), $(1.0),
        $(Val(cumdis)), $(Val(area_mode)), $(Val(grad)))
end
