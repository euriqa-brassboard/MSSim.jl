#!/usr/bin/julia

using MSSim
using BenchmarkTools

const U = MSSim.Utils
const SL = MSSim.SymLinear
const SS = MSSim.SegSeq

for (cumdis, area_mode, grad) in Iterators.product((false, true), (false, true),
                                                   (false, true))
    @show (cumdis, area_mode, grad)
    T = Float64
    CT = Complex{T}
    D = CT
    A = T
    CD = cumdis ? CT : Nothing
    DG = area_mode ? CT : Nothing
    AG = area_mode ? T : Nothing
    grads = grad ? U.JaggedMatrix{SS.SegData{T,D,A,CD,DG,AG}}() : nothing
    function compute_values()
        res, g = SL.SegInt.compute_values(1.0, 1.0, 1.0, 1.0, 1.0,
                                          Val(cumdis), Val(area_mode), Val(grad))
        grad && push!(grads, g)
        return res
    end
    segs = [compute_values() for i in 1:100]

    result = SS.SingleModeResult{T,D,A,CD,DG,AG}()
    buffer = SS.SeqComputeBuffer{T}()

    @btime SS.compute_single_mode!($result, $segs, $buffer, $grads)
end
