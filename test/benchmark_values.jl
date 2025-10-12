#!/usr/bin/julia

using MSSim: Utils as U, SymLinear as SL, SegSeq as SS
using BenchmarkTools

for (cumdis, area_mode, grad) in Iterators.product((false, true), (false, true),
                                                   (false, true))
    @show (cumdis, area_mode, grad)
    maskv = SS.ValueMask(true, true, true, cumdis, area_mode, area_mode)
    maskg = grad ? maskv : zero(SS.ValueMask)
    @btime SL.SegInt.compute_values(
        $(1.0), $(1.0), $(1.0), $(1.0), $(1.0),
        $(Val(maskv)), $(Val(maskg)))
    @btime SL.SegInt.compute_values(
        $(1.0), $(1.0), $(U.Zero()), $(1.0), $(1.0),
        $(Val(maskv)), $(Val(maskg)))
    if grad
        @btime begin
            v, g = SL.SegInt.compute_values(
                $(1.0), $(1.0), $(1.0), $(1.0), $(1.0),
                $(Val(maskv)), $(Val(maskg)))
            v, g[5]
        end
        @btime begin
            v, g = SL.SegInt.compute_values(
                $(1.0), $(1.0), $(U.Zero()), $(1.0), $(1.0),
                $(Val(maskv)), $(Val(maskg)))
            v, g[5]
        end
    end
end
