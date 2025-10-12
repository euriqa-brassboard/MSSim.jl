#!/usr/bin/julia

using MSSim: Utils as U, SymLinear as SL, SegSeq as SS
using BenchmarkTools

for (cumdis, area_mode, grad) in Iterators.product((false, true), (false, true),
                                                   (false, true))
    @show (cumdis, area_mode, grad)
    maskv = SS.ValueMask(true, true, true, cumdis, area_mode, area_mode)
    maskg = grad ? maskv : zero(SS.ValueMask)

    T = Float64
    SDV = SS.SegData(T, maskv)
    SDG = SS.SegData(T, maskg)
    grads = grad ? U.JaggedMatrix{SDG}() : nothing
    function compute_values()
        res, g = SL.SegInt.compute_values(1.0, 1.0, 1.0, 1.0, 1.0,
                                          Val(maskv), Val(maskg))
        grad && push!(grads, g)
        return res
    end
    segs = [compute_values() for i in 1:100]

    result = SS.SingleModeResult{T}(Val(maskv), Val(maskg))
    buffer = SS.SeqComputeBuffer{T}()

    @btime SS.compute_single_mode!($result, $segs, $buffer, $grads)
end
