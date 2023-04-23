#!/usr/bin/julia

using MSSim
using BenchmarkTools

for (cumdis, area_mode, grad) in Iterators.product((false, true), (false, true),
                                                   (false, true))
    @show (cumdis, area_mode, grad)
    maskv = MSSim.SegSeq.ValueMask(true, true, true, cumdis, area_mode, area_mode)
    maskg = grad ? maskv : zero(MSSim.SegSeq.ValueMask)
    @btime MSSim.SymLinear.SegInt.compute_values(
        $(1.0), $(1.0), $(1.0), $(1.0), $(1.0),
        $(Val(maskv)), $(Val(maskg)))
    @btime MSSim.SymLinear.SegInt.compute_values(
        $(1.0), $(1.0), $(MSSim.Utils.Zero()), $(1.0), $(1.0),
        $(Val(maskv)), $(Val(maskg)))
    if grad
        @btime begin
            v, g = MSSim.SymLinear.SegInt.compute_values(
                $(1.0), $(1.0), $(1.0), $(1.0), $(1.0),
                $(Val(maskv)), $(Val(maskg)))
            v, g[5]
        end
        @btime begin
            v, g = MSSim.SymLinear.SegInt.compute_values(
                $(1.0), $(1.0), $(MSSim.Utils.Zero()), $(1.0), $(1.0),
                $(Val(maskv)), $(Val(maskg)))
            v, g[5]
        end
    end
end
