#!/usr/bin/julia

module Optimizers

import ForwardDiff
using StaticArrays

import ..Sequence as Seq
import ..SymLinear as SL
import ..SegSeq as SS

struct NLVarTracker
    vars::Vector{Tuple{Float64,Float64}}
    NLVarTracker(nargs) = new(fill((-Inf, Inf), nargs))
end

function set_bound!(tracker::NLVarTracker, idx, lb, ub)
    tracker.vars[idx] = (lb, ub)
    return
end

lower_bounds(tracker::NLVarTracker) = [lb for (lb, ub) in tracker.vars]
upper_bounds(tracker::NLVarTracker) = [ub for (lb, ub) in tracker.vars]

function init_vars!(tracker::NLVarTracker, vars=nothing)
    nvars = length(tracker.vars)
    if vars === nothing
        vars = Vector{Float64}(undef, nvars)
    end
    for vi in 1:nvars
        lb, ub = tracker.vars[vi]
        if isfinite(lb)
            if isfinite(ub)
                vars[vi] = lb + (ub - lb) * rand()
            else
                vars[vi] = lb + rand()
            end
        elseif isfinite(ub)
            vars[vi] = ub - rand()
        else
            vars[vi] = rand()
        end
    end
    return vars
end

function autodiff(f::F) where F
    function fn_with_diff(x, grad)
        if !isempty(grad)
            # Use ForwardDiff to compute the gradient. Replace with your
            # favorite Julia automatic differentiation package.
            ForwardDiff.gradient!(grad, f, x)
        end
        return f(x)
    end
end

struct AbsAreaObjCallback{NModes}
    dis_weights::MVector{NModes,Float64}
    disδ_weights::MVector{NModes,Float64}
    area_weights::MVector{NModes,Float64}
end

function (obj::AbsAreaObjCallback{NModes})(vals, grads) where NModes
    @assert length(vals) == 3 * NModes
    @assert length(grads) == 3 * NModes
    v1 = zero(eltype(vals))
    v2 = zero(eltype(vals))
    @inbounds @simd ivdep for i in 1:NModes
        v1 = muladd(vals[i], obj.dis_weights[i],
                    muladd(vals[i + NModes], obj.disδ_weights[i], v1))
        v2 = muladd(abs(vals[i + NModes * 2]), obj.area_weights[i], v2)
    end
    iv2 = 1 / v2

    res = v1 * iv2
    dv1 = iv2
    dv2 = -res * iv2

    @inbounds @simd ivdep for i in 1:NModes
        grads[i] = obj.dis_weights[i] * dv1
        grads[i + NModes] = obj.disδ_weights[i] * dv1
        grads[i + NModes * 2] = flipsign(obj.area_weights[i] * dv2, vals[i + NModes * 2])
    end
    return res
end

function _init_weights!(weights, input)
    if input === nothing
        weights .= 1
    else
        weights .= input
    end
    return
end

function abs_area_obj(nseg, modes, pmask;
                      freq=Seq.FreqSpec(), amp=Seq.AmpSpec(),
                      dis_weights=nothing, disδ_weights=nothing, area_weights=nothing)
    nmodes = length(modes.modes)
    dis_args = ntuple(i->(:dis2, i), nmodes)
    disδ_args = ntuple(i->(:disδ2, i), nmodes)
    area_args = ntuple(i->(:area, i), nmodes)

    cb = AbsAreaObjCallback{nmodes}(MVector{nmodes,Float64}(undef),
                                    MVector{nmodes,Float64}(undef),
                                    MVector{nmodes,Float64}(undef))
    _init_weights!(cb.dis_weights, dis_weights)
    _init_weights!(cb.disδ_weights, disδ_weights)
    _init_weights!(cb.area_weights, area_weights)

    mask_dis_area = SS.ValueMask(true, true, true, false, true, false)
    buf = SL.ComputeBuffer{nseg,Float64}(Val(mask_dis_area), Val(mask_dis_area))

    return Seq.Objective(pmask, (dis_args..., disδ_args..., area_args...),
                         cb, modes, buf, freq=freq, amp=amp)
end

end
