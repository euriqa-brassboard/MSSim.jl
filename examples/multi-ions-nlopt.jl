#!/usr/bin/julia

push!(LOAD_PATH, joinpath(@__DIR__, "../../lib"))

using MSSim
using BenchmarkTools

const Opts = MSSim.Optimizers
const SS = MSSim.SegSeq
const SL = MSSim.SymLinear
const Seq = MSSim.Sequence

using NLopt
using StaticArrays

nseg = 120
buf = SL.ComputeBuffer{nseg,Float64}(Val(SS.mask_allδ), Val(SS.mask_allδ))
# buf = SL.ComputeBuffer{nseg,Float64}(Val(SS.mask_full), Val(SS.mask_full))

η_Yb171(f) = 0.1924385393508112 / sqrt(f)

const fs = [2.353762760667294, 2.4012136493819183, 2.443017513112808, 2.479035422108152, 2.5090222263685185, 2.5327427588303233, 2.5494]
const ηs = η_Yb171.(fs)
const bij = [0.022206965694101886 -0.17297536129705904 0.4902047215008256 -0.6784055713274825 0.48884305881109147 -0.1720269956607397 0.022153182279263953
             -0.08463441105336124 0.4122403648898327 -0.5681256075100114 -0.0014096657470350112 0.5687687355176414 -0.41162962776905704 0.08479021167198833
             -0.21164031485188461 0.570551914643521 -0.11994168757151538 -0.477744541573237 -0.12027067168024524 0.5720345839925919 -0.212989282959229
             0.39446370843399914 -0.44596812195515206 -0.3810325372909801 0.0009952934725295946 0.38220272065861055 0.4448051290678067 -0.39546619238680936
             0.5586475771790227 -0.032598428141922896 -0.32134292250975366 -0.41068955709650196 -0.3207887698322381 -0.03095489420589502 0.5577269946072854
             0.580522496250635 0.36304345808950844 0.17675521353985915 2.9273378799938786e-5 -0.17671462162762466 -0.36342433304004457 -0.5802114865911353
             -0.3779644730092277 -0.3779644730092278 -0.37796447300922703 -0.3779644730092274 -0.37796447300922614 -0.3779644730092255 -0.3779644730092287]

function mode_weight!(weights, ion1, ion2)
    for i in 1:7
        weights[i] = bij[i, ion1] * bij[i, ion2] * ηs[i]^2
    end
end

const ion1 = 2
const ion2 = 6

const modes = Seq.Modes()
for i in 1:7
    f = fs[i]
    push!(modes, 2π * f, 1)
end

struct PreOptObjective{NModes}
end

function (::PreOptObjective{NModes})(vals) where NModes
    dis = vals[1]
    disδ = vals[2]
    area = zero(eltype(vals))
    areaδ = zero(eltype(vals))
    for i in 1:NModes
        areai = abs(vals[2 + i])
        areaδi = abs(vals[2 + NModes + i])
        area += areai
        areaδ += areaδi / areai
    end
    return (dis + disδ / 200) * areaδ / area^2
end

struct ARobustObjective{NModes}
    weights::MVector{NModes,Float64}
    ARobustObjective{NModes}(weights=MVector{NModes,Float64}(undef)) where NModes =
        new(weights)
end

function (obj::ARobustObjective{NModes})(vals) where NModes
    dis = vals[1]
    disδ = vals[2]
    area = zero(eltype(vals))
    areaδ = zero(eltype(vals))
    for i in 1:NModes
        weight = obj.weights[i]
        areai = vals[2 + i] * weight
        areaδi = vals[2 + NModes + i] * weight
        area += areai
        areaδ += areaδi
    end
    return 5 * dis + disδ / 100 + (abs(area) - π / 2)^2 * 100 + (areaδ / 1e4)^2
end

blackman(x) = 0.42 + 0.5 * cospi(x) + 0.08 * cospi(2 * x)

struct BlackmanStartEnd{Ratio} <: Function
end

function (::BlackmanStartEnd{Ratio})(x) where Ratio
    if -Ratio <= x <= Ratio
        return float(one(x))
    elseif x < 0
        x = (x + Ratio) / (1 - Ratio)
    else
        x = (x - Ratio) / (1 - Ratio)
    end
    return blackman(x)
end

const preobj = Seq.Objective(SL.pmask_tfm,
                              ((:dis2, 0), (:disδ2, 0),
                               (:area, 1), (:area, 2), (:area, 3), (:area, 4),
                               (:area, 5), (:area, 6), (:area, 7),
                               (:areaδ, 1), (:areaδ, 2), (:areaδ, 3), (:areaδ, 4),
                               (:areaδ, 5), (:areaδ, 6), (:areaδ, 7)),
                              Opts.autodiff(PreOptObjective{7}()), modes, buf,
                              freq=Seq.FreqSpec(true, sym=false),
                              amp=Seq.AmpSpec(cb=BlackmanStartEnd{0.6}(), mid_order=-1))

struct PreOptimizer{Obj}
    preobj::Obj
    tracker::Opts.NLVarTracker
    opt::NLopt.Opt
end
function PreOptimizer(preobj::Obj) where Obj
    nargs = Seq.nparams(preobj)
    tracker = Opts.NLVarTracker(nargs)
    Opts.set_bound!(tracker, preobj.param.τ, 0.1, 2)
    Opts.set_bound!(tracker, preobj.param.Ωbase, 0.7, 0.7)
    for ω in preobj.param.ωs
        Opts.set_bound!(tracker, ω, 2π * 2.39, 2π * 2.52)
    end

    opt = NLopt.Opt(:LD_LBFGS, nargs)
    NLopt.min_objective!(opt, preobj)
    NLopt.lower_bounds!(opt, Opts.lower_bounds(tracker))
    NLopt.upper_bounds!(opt, Opts.upper_bounds(tracker))
    NLopt.maxeval!(opt, 2000)

    return PreOptimizer{Obj}(preobj, tracker, opt)
end
function compute_one(o::PreOptimizer)
    obj, params, ret = NLopt.optimize(o.opt, Opts.init_vars!(o.tracker))
    if getfield(NLopt, ret) < 0
        return
    end
    total_t = o.preobj(Val((:τ, 0)), params)
    dis = o.preobj(Val((:dis2, 0)), params)
    disδ = o.preobj(Val((:disδ2, 0)), params)
    area = 0.0
    areaδ = 0.0
    for i in 1:7
        areai = abs(o.preobj(Val((:area, i)), params))
        areaδi = abs(o.preobj(Val((:areaδ, i)), params))
        area += areai
        areaδ += areaδi / areai
    end
    # @show obj, dis, disδ, area, areaδ, total_t, params[preobj.param.Ωbase]
    if abs(dis) < 1e-6 && abs(disδ) < 1e-4 && area >= 100
        return params
    end
end

const preopt = PreOptimizer(preobj)

function pre_optimize!(preopt, nrounds, candidates=Vector{Float64}[])
    for i in 1:nrounds
        params = compute_one(preopt)
        if params !== nothing
            push!(candidates, params)
        end
    end
    return candidates
end

const arobust_kernel = ARobustObjective{7}()
const arobust_obj = Seq.Objective(SL.pmask_full,
                                  ((:dis2, 0), (:disδ2, 0),
                                   (:area, 1), (:area, 2), (:area, 3), (:area, 4),
                                   (:area, 5), (:area, 6), (:area, 7),
                                   (:areaδ, 1), (:areaδ, 2), (:areaδ, 3), (:areaδ, 4),
                                   (:areaδ, 5), (:areaδ, 6), (:areaδ, 7)),
                                  Opts.autodiff(arobust_kernel), modes, buf,
                                  freq=Seq.FreqSpec(true, sym=false),
                                  amp=Seq.AmpSpec(cb=BlackmanStartEnd{0.6}(),
                                                  mid_order=-1))

struct ParamInfo
    params::Vector{Float64}
    dis::Float64
    disδ::Float64
    area::Float64
    areaδ::Float64
    total_t::Float64
    maxΩ::Float64
end

function optimize_pairs(kernel, obj, preopt)
    function compute_properties(params)
        total_t = obj(Val((:τ, 0)), params)
        dis = obj(Val((:dis2, 0)), params)
        disδ = obj(Val((:disδ2, 0)), params)
        area = 0.0
        areaδ = 0.0
        for i in 1:7
            weight = kernel.weights[i]
            areai = obj(Val((:area, i)), params) * weight
            areaδi = obj(Val((:areaδ, i)), params) * weight
            area += areai
            areaδ += areaδi
        end
        return dis, disδ, area, areaδ, total_t, params[obj.param.Ωbase]
    end
    pairs = Set{Tuple{Int,Int}}()
    for ion1 in 3:6
        for ion2 in 2:ion1 - 1
            push!(pairs, (ion1, ion2))
        end
    end


    ####################
    delete!(pairs, (5, 3))
    delete!(pairs, (5, 4))
    delete!(pairs, (4, 3))
    ####################

    candidates_checked = 0
    function check_modes(candidates_checked)
        for (ion1, ion2) in collect(pairs)
            mode_weight!(kernel.weights, ion1, ion2)
            max_area = 0.0
            found = false
            for param_idx in candidates_checked + 1:length(candidates)
                params = candidates[param_idx]
                dis, disδ, area, areaδ, total_t, maxΩ = compute_properties(params)
                max_area = max(max_area, abs(area))
                if abs(area) >= 1.56
                    println("Accept $(abs(area)) for $(ion1), $(ion2)")
                    found = true
                    delete!(pairs, (ion1, ion2))
                    break
                end
            end
            if !found
                println("Reject $(max_area) for $(ion1), $(ion2)")
            end
        end
        return length(candidates)
    end
    candidates = Vector{Float64}[]
    @time pre_optimize!(preopt, 500, candidates)
    while !isempty(pairs)
        @time pre_optimize!(preopt, 100, candidates)
        candidates_checked = check_modes(candidates_checked)
        @show candidates_checked, pairs
    end
    pre_optimize!(preopt, 500, candidates)

    nargs = Seq.nparams(obj)
    tracker = Opts.NLVarTracker(nargs)
    Opts.set_bound!(tracker, obj.param.τ, 0.1, 2)
    Opts.set_bound!(tracker, obj.param.Ωbase, 0.01, 0.71)
    for ω in obj.param.ωs
        Opts.set_bound!(tracker, ω, 2π * 2.39, 2π * 2.52)
    end

    # opt = NLopt.Opt(:LD_SLSQP, nargs) # 13 ms
    opt = NLopt.Opt(:LD_LBFGS, nargs) # 50 ms
    # NLopt.xtol_rel!(opt, 1e-7)
    # NLopt.ftol_rel!(opt, 1e-7)
    NLopt.min_objective!(opt, obj)
    NLopt.lower_bounds!(opt, Opts.lower_bounds(tracker))
    NLopt.upper_bounds!(opt, Opts.upper_bounds(tracker))
    NLopt.maxeval!(opt, 10000)
    for ion1 in 3:6
        for ion2 in 2:ion1 - 1
            ####################
            if (ion1, ion2) in ((5, 3), (5, 4), (4, 3))
                continue
            end
            ####################



            mode_weight!(kernel.weights, ion1, ion2)
            tier1 = ParamInfo[]
            tier2 = ParamInfo[]
            tier3 = ParamInfo[]
            tier4 = ParamInfo[]
            for params in candidates
                dis, disδ, area, areaδ, total_t, maxΩ = compute_properties(params)
                if abs(area) < 1.56
                    continue
                end
                info = ParamInfo(params, dis, disδ, area, areaδ, total_t, maxΩ)
                if abs(areaδ) < abs(area) * 2 && abs(area) > 1.8 && total_t <= 220
                    push!(tier1, info)
                elseif abs(areaδ) < abs(area) * 2 && (abs(area) > 1.8 || total_t <= 220)
                    push!(tier2, info)
                elseif abs(area) > 1.8 || total_t <= 220
                    push!(tier3, info)
                else
                    push!(tier4, info)
                end
            end
            if !isempty(tier1)
                mode_cand = tier1
            elseif !isempty(tier2)
                mode_cand = tier2
            elseif !isempty(tier3)
                mode_cand = tier3
            else
                @assert !isempty(tier4)
                mode_cand = tier4
            end
            println("Found $(length(mode_cand)) candidates for $ion1, $ion2")
            # sort!(mode_cand, by=x->(abs(x.areaδ / x.area), 1 / abs(x.area), x.total_t))

            min_areaδ = 0.0
            max_area = 0.0
            min_total_t = 0.0
            for info in mode_cand
                areaδ = abs(info.areaδ / info.area)
                area = abs(info.area)
                total_t = info.total_t
                min_areaδ = min(areaδ, min_areaδ)
                max_area = max(area, max_area)
                min_total_t = min(total_t, min_total_t)
            end
            tier1 = ParamInfo[]
            tier2 = ParamInfo[]
            tier3 = ParamInfo[]
            tier4 = ParamInfo[]
            for info in mode_cand
                areaδ = abs(info.areaδ / info.area)
                area = abs(info.area)
                total_t = info.total_t
                if (areaδ < min_areaδ * 3 &&
                    (area > max_area * 0.7 && total_t <= min_total_t + 20))
                    push!(tier1, info)
                elseif (areaδ < min_areaδ * 3 &&
                    (area > max_area * 0.7 || total_t <= min_total_t + 20))
                    push!(tier2, info)
                elseif area > max_area * 0.7 || total_t <= min_total_t + 20
                    push!(tier3, info)
                else
                    push!(tier4, info)
                end
            end
            if !isempty(tier1)
                mode_cand = tier1
            elseif !isempty(tier2)
                mode_cand = tier2
            elseif !isempty(tier3)
                mode_cand = tier3
            else
                @assert !isempty(tier4)
                mode_cand = tier4
            end
            println("Found $(length(mode_cand)) candidates for $ion1, $ion2")

            tier1 = ParamInfo[]
            tier2 = ParamInfo[]
            tier3 = ParamInfo[]
            tier4 = ParamInfo[]
            for old_info in mode_cand
                params = copy(old_info.params)
                params[obj.param.Ωbase] *= sqrt(π / 2 / abs(old_info.area))
                objv, params, ret = NLopt.optimize!(opt, params)
                dis, disδ, area, areaδ, total_t, maxΩ = compute_properties(params)
                info = ParamInfo(params, dis, disδ, area, areaδ, total_t, maxΩ)
                if abs(areaδ) < abs(area) * 1.5 && maxΩ < 0.6 && total_t <= 220
                    push!(tier1, info)
                elseif abs(areaδ) < abs(area) * 1.5 && (maxΩ < 0.6 || total_t <= 220)
                    push!(tier2, info)
                elseif maxΩ < 0.6 || total_t <= 220
                    push!(tier3, info)
                else
                    push!(tier4, info)
                end
            end
            if !isempty(tier1)
                mode_cand = tier1
            elseif !isempty(tier2)
                mode_cand = tier2
            elseif !isempty(tier3)
                mode_cand = tier3
            else
                @assert !isempty(tier4)
                mode_cand = tier4
            end
            println("Found $(length(mode_cand)) candidates for $ion1, $ion2")
            sort!(mode_cand, by=x->(abs(x.areaδ / x.area), 1 / abs(x.area), x.total_t))
            info0 = mode_cand[1]
            @show info0.dis, info0.disδ, info0.area, info0.areaδ, info0.total_t, info0.maxΩ
        end
    end
end
optimize_pairs(arobust_kernel, arobust_obj, preopt)

# const candidates = Vector{Float64}[]
# @time for i in 1:1000
#     obj, params, ret = NLopt.optimize(opt, Opts.init_vars!(preopt_tracker))
#     if getfield(NLopt, ret) < 0
#         continue
#     end
#     total_t = preobj(Val((:τ, 0)), params)
#     dis = preobj(Val((:dis2, 0)), params)
#     disδ = preobj(Val((:disδ2, 0)), params)
#     area = 0.0
#     areaδ = 0.0
#     for i in 1:7
#         areai = abs(preobj(Val((:area, i)), params))
#         areaδi = abs(preobj(Val((:areaδ, i)), params))
#         area += areai
#         areaδ += areaδi / areai
#     end
#     if abs(dis) < 1e-6 && abs(disδ) < 1e-5 && area >= 100
#         push!(candidates, params)
#         @show obj, dis, disδ, area, areaδ, total_t, params[preobj.param.Ωbase]
#     end
# end
# @show length(candidates)
