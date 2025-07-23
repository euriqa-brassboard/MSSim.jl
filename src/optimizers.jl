#!/usr/bin/julia

module Optimizers

import ForwardDiff
using StaticArrays

import ..Sequence as Seq
import ..SymLinear as SL
import ..SegSeq as SS
import ..Utils as U

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

function _init_weights!(weights, input)
    if input === nothing
        weights .= 1
    else
        weights .= input
    end
    return
end

struct AbsAreaObjCallback{NW}
    weights::MVector{NW,Float64}
    function AbsAreaObjCallback(NModes, dis_weights, disδ_weights, area_weights)
        @assert NModes >= 1
        cb = new{NModes * 3}(MVector{NModes * 3,Float64}(undef))
        _init_weights!(cb.dis_weights, dis_weights)
        _init_weights!(cb.disδ_weights, disδ_weights)
        _init_weights!(cb.area_weights, area_weights)
        return cb
    end
end

@inline function Base.getproperty(obj::AbsAreaObjCallback{NW}, field::Symbol) where NW
    NModes = NW ÷ 3
    w = getfield(obj, :weights)
    if field === :dis_weights
        return @view w[1:NModes]
    elseif field === :disδ_weights
        return @view w[NModes + 1:2 * NModes]
    elseif field === :area_weights
        return @view w[2 * NModes + 1:3 * NModes]
    else
        return getfield(obj, field)
    end
end

function (obj::AbsAreaObjCallback{NW})(vals, grads) where NW
    @assert length(vals) == NW
    @assert length(grads) == NW
    NModes = NW ÷ 3
    weights = obj.weights
    @inbounds begin
        v1 = muladd(vals[1], weights[1], U.mul(vals[1 + NModes], weights[1 + NModes]))
        v2 = U.mul(abs(vals[1 + NModes * 2]), weights[1 + NModes * 2])
    end
    @inbounds @simd ivdep for i in 2:NModes
        v1 = muladd(vals[i], weights[i],
                    muladd(vals[i + NModes], weights[i + NModes], v1))
        v2 = muladd(abs(vals[i + NModes * 2]), weights[i + NModes * 2], v2)
    end
    iv2 = 1 / v2

    res = v1 * iv2
    dv1 = iv2
    dv2 = -res * iv2

    @inbounds @simd ivdep for i in 1:NModes
        grads[i] = weights[i] * dv1
        grads[i + NModes] = weights[i + NModes] * dv1
        grads[i + NModes * 2] = flipsign(weights[i + NModes * 2] * dv2,
                                         vals[i + NModes * 2])
    end
    return res
end

function abs_area_obj(nseg, modes, pmask;
                      freq=Seq.FreqSpec(), amp=Seq.AmpSpec(),
                      dis_weights=nothing, disδ_weights=nothing, area_weights=nothing)
    nmodes = length(modes.modes)
    dis_args = ntuple(i->(:dis2, i), nmodes)
    disδ_args = ntuple(i->(:disδ2, i), nmodes)
    area_args = ntuple(i->(:area, i), nmodes)

    mask_dis_area = SS.ValueMask(true, true, true, false, true, false)
    buf = SL.ComputeBuffer{nseg,Float64}(Val(mask_dis_area), Val(mask_dis_area))

    return Seq.Objective(pmask, (dis_args..., disδ_args..., area_args...),
                         AbsAreaObjCallback(nmodes, dis_weights, disδ_weights,
                                            area_weights),
                         modes, buf, freq=freq, amp=amp)
end

struct AreaTarget
    start_idx::Int
    target::Float64
    area_weights::Vector{Float64}
    areaδ_weights::Vector{Float64}
    function AreaTarget(start_idx; target=π / 2, area_weights, areaδ_weights=Float64[])
        return new(start_idx, target, area_weights, areaδ_weights)
    end
end

struct _AreaInfo
    mode_start_idx::Int
    nmodes::Int
    target_idx::Int # area_weight index is target_idx + 1
    has_areaδ::Bool
end

# AreaOffsets: tuple of _AreaInfo
struct TargetAreaObjCallback{NP,NModes,AreaOffsets}
    params::MVector{NP,Float64}
    function TargetAreaObjCallback(NModes, dis_weights, disδ_weights, area_targets)
        @assert NModes >= 1

        params = Vector{Float64}(undef, 2 * NModes)
        _init_weights!(@view(params[1:NModes]), dis_weights)
        _init_weights!(@view(params[NModes + 1:2 * NModes]), disδ_weights)

        area_info = _AreaInfo[]
        area_idx_offset = 2 * NModes
        for area_tgt::AreaTarget in area_targets
            @assert area_tgt.start_idx <= NModes
            area_nmodes = length(area_tgt.area_weights)
            @assert area_tgt.start_idx + area_nmodes - 1 <= NModes
            has_areaδ = !isempty(area_tgt.areaδ_weights)
            target_idx = area_idx_offset + 1
            push!(area_info, _AreaInfo(area_tgt.start_idx, area_nmodes,
                                       target_idx, has_areaδ))
            area_idx_offset += has_areaδ ? (2 * area_nmodes + 1) : (area_nmodes + 1)
            resize!(params, area_idx_offset)
            params[target_idx] = area_tgt.target
            params[target_idx + 1:target_idx + area_nmodes] .= area_tgt.area_weights
            if has_areaδ
                @assert length(area_tgt.areaδ_weights) == area_nmodes
                params[target_idx + 1 + area_nmodes:target_idx + 2 * area_nmodes] .=
                    area_tgt.areaδ_weights
            end
        end
        NP = length(params)
        return new{NP,NModes,(area_info...,)}(params)
    end
end

struct AreaTargetWrapper{Obj,AreaInfo}
    _obj::Obj
end
@inline function Base.getproperty(wrapper::AreaTargetWrapper{Obj,AreaInfo},
                                  field::Symbol) where {Obj,AreaInfo}
    p = getfield(wrapper, :_obj).params
    if field === :target
        return p[AreaInfo.target_idx]
    elseif field === :area_weights
        return @view p[AreaInfo.target_idx + 1:AreaInfo.target_idx + AreaInfo.nmodes]
    elseif field === :areaδ_weights
        if AreaInfo.has_areaδ
            return @view p[AreaInfo.target_idx + AreaInfo.nmodes + 1:AreaInfo.target_idx + AreaInfo.nmodes * 2]
        else
            return nothing
        end
    else
        return getfield(wrapper, field)
    end
end
@inline function Base.setproperty!(wrapper::AreaTargetWrapper{Obj,AreaInfo},
                                   field::Symbol, value) where {Obj,AreaInfo}
    p = getfield(wrapper, :_obj).params
    if field === :target
        p[AreaInfo.target_idx] = value
    else
        setfield!(wrapper, field, value)
    end
    return
end

@inline function Base.getproperty(obj::TargetAreaObjCallback{NP,NModes,AreaOffsets}, field::Symbol) where {NP,NModes,AreaOffsets}
    p = getfield(obj, :params)
    if field === :dis_weights
        return @view p[1:NModes]
    elseif field === :disδ_weights
        return @view p[NModes + 1:2 * NModes]
    elseif field === :area_targets
        return ntuple(i->AreaTargetWrapper{typeof(obj),AreaOffsets[i]}(obj),
                      length(AreaOffsets))
    else
        return getfield(obj, field)
    end
end

@generated function (obj::TargetAreaObjCallback{NP,NModes,AreaOffsets})(vals, grads) where {NP,NModes,AreaOffsets}
    func_ex = quote
        @assert length(vals) == $(NModes * 4)
        @assert length(grads) == $(NModes * 4)
        params = getfield(obj, :params)
        res = zero(eltype(vals))
        @inbounds @simd ivdep for i in 1:$(2 * NModes)
            res = muladd(vals[i], params[i], res)
            grads[i] = params[i]
        end
    end
    area_used = [Int[] for i in 1:NModes]
    areaδ_used = [Int[] for i in 1:NModes]
    for (info_i, area_info) in enumerate(AreaOffsets)
        mode_start_idx = area_info.mode_start_idx
        nmodes = area_info.nmodes
        for mode_i in mode_start_idx:mode_start_idx + nmodes - 1
            push!(area_used[mode_i], info_i)
            if area_info.has_areaδ
                push!(areaδ_used[mode_i], info_i)
            end
        end
    end
    for mode_i in 1:NModes
        if isempty(area_used[mode_i])
            push!(func_ex.args, :(@inbounds grads[$(mode_i + NModes * 2)] = 0))
        end
        if isempty(areaδ_used[mode_i])
            push!(func_ex.args, :(@inbounds grads[$(mode_i + NModes * 3)] = 0))
        end
    end
    for (info_i, area_info) in enumerate(AreaOffsets)
        mode_start_idx = area_info.mode_start_idx
        nmodes = area_info.nmodes
        target_idx = area_info.target_idx
        has_areaδ = area_info.has_areaδ
        push!(func_ex.args, quote
                  @inbounds begin
                      area = U.mul(vals[$(NModes * 2 + mode_start_idx)],
                                   params[$(target_idx + 1)])
                      $(has_areaδ ? quote
                            areaδ = U.mul(vals[$(NModes * 3 + mode_start_idx)],
                                           params[$(target_idx + nmodes + 1)])
                        end : nothing)
                  end
                  @inbounds @simd ivdep for i in 2:$nmodes
                      area = muladd(vals[$(NModes * 2 + mode_start_idx - 1) + i],
                                    params[$target_idx + i], area)
                      $(has_areaδ ? quote
                            areaδ = muladd(vals[$(NModes * 3 + mode_start_idx - 1) + i],
                                            params[$(target_idx + nmodes) + i], areaδ)
                        end : nothing)
                  end
                  area_err = abs(area) - @inbounds params[$target_idx]
                  area_err2 = flipsign(area_err * 2, area)
                  res = muladd(area_err, area_err, res)
                  $(has_areaδ ? quote
                        areaδ2 = areaδ * 2
                        res = muladd(areaδ, areaδ, res)
                    end : nothing)
              end)

        area_all_first = true
        areaδ_all_first = true
        for mode_i in mode_start_idx:mode_start_idx + nmodes - 1
            if area_used[mode_i][1] != info_i
                area_all_first = false
            end
            if area_info.has_areaδ
                if areaδ_used[mode_i][1] != info_i
                    areaδ_all_first = false
                end
            end
        end

        if area_all_first
            push!(func_ex.args, quote
                      @inbounds @simd ivdep for i in 1:$nmodes
                          grads[$(NModes * 2 + mode_start_idx - 1) + i] =
                              params[$target_idx + i] * area_err2
                      end
                  end)
        else
            for mode_i in mode_start_idx:mode_start_idx + nmodes - 1
                grad_idx = NModes * 2 + mode_i
                param_idx = target_idx + 1 + mode_i - mode_start_idx
                if area_used[mode_i][1] == info_i
                    push!(func_ex.args,
                          :(@inbounds(grads[$grad_idx] = params[$param_idx] * area_err2)))
                else
                    push!(func_ex.args,
                          :(@inbounds(grads[$grad_idx] =
                              muladd(params[$param_idx], area_err2, grads[$grad_idx]))))
                end
            end
        end

        if !has_areaδ
            continue
        end
        if areaδ_all_first
            push!(func_ex.args, quote
                      @inbounds @simd ivdep for i in 1:$nmodes
                          grads[$(NModes * 3 + mode_start_idx - 1) + i] =
                              params[$(target_idx + nmodes) + i] * areaδ2
                      end
                  end)
        else
            for mode_i in mode_start_idx:mode_start_idx + nmodes - 1
                grad_idx = NModes * 3 + mode_i
                param_idx = target_idx + nmodes + 1 + mode_i - mode_start_idx
                if area_used[mode_i][1] == info_i
                    push!(func_ex.args,
                          :(@inbounds(grads[$grad_idx] = params[$param_idx] * areaδ2)))
                else
                    push!(func_ex.args,
                          :(@inbounds(grads[$grad_idx] =
                              muladd(params[$param_idx], areaδ2, grads[$grad_idx]))))
                end
            end
        end
    end
    push!(func_ex.args, quote
              return res
          end)
    return func_ex
end

function target_area_obj(nseg, modes, pmask;
                         freq=Seq.FreqSpec(), amp=Seq.AmpSpec(),
                         dis_weights=nothing, disδ_weights=nothing, area_targets)
    nmodes = length(modes.modes)
    dis_args = ntuple(i->(:dis2, i), nmodes)
    disδ_args = ntuple(i->(:disδ2, i), nmodes)
    area_args = ntuple(i->(:area, i), nmodes)
    areaδ_args = ntuple(i->(:areaδ, i), nmodes)

    val_mask = SS.ValueMask(true, true, true, false, true, true)
    buf = SL.ComputeBuffer{nseg,Float64}(Val(val_mask), Val(val_mask))

    return Seq.Objective(pmask, (dis_args..., disδ_args..., area_args...),
                         TargetAreaObjCallback(nmodes, dis_weights, disδ_weights,
                                               area_targets),
                         modes, buf, freq=freq, amp=amp)
end

end
