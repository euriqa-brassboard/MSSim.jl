#!/usr/bin/julia

push!(LOAD_PATH, joinpath(@__DIR__, "../../lib"))

using MSSim
using BenchmarkTools

const Opts = MSSim.Optimizers
const SS = MSSim.SegSeq
const SL = MSSim.SymLinear
const Seq = MSSim.Sequence

using NLopt

nseg = 100
buf = SL.ComputeBuffer{nseg,Float64}(Val(SS.mask_allδ), Val(SS.mask_allδ))
# buf = SL.ComputeBuffer{nseg,Float64}(Val(SS.mask_full), Val(SS.mask_full))

const fs = [2.353762760667294, 2.4012136493819183, 2.443017513112808, 2.479035422108152, 2.5090222263685185, 2.5327427588303233, 2.5494]
const bij = [0.022206965694101886 -0.08463441105336124 -0.21164031485188461 0.39446370843399914 0.5586475771790227 0.580522496250635 -0.3779644730092277
             -0.17297536129705904 0.4122403648898327 0.570551914643521 -0.44596812195515206 -0.032598428141922896 0.36304345808950844 -0.3779644730092278
             0.4902047215008256 -0.5681256075100114 -0.11994168757151538 -0.3810325372909801 -0.32134292250975366 0.17675521353985915 -0.37796447300922703
             -0.6784055713274825 -0.0014096657470350112 -0.477744541573237 0.0009952934725295946 -0.41068955709650196 2.9273378799938786e-5 -0.3779644730092274
             0.48884305881109147 0.5687687355176414 -0.12027067168024524 0.38220272065861055 -0.3207887698322381 -0.17671462162762466 -0.37796447300922614
             -0.1720269956607397 -0.41162962776905704 0.5720345839925919 0.4448051290678067 -0.03095489420589502 -0.36342433304004457 -0.3779644730092255
             0.022153182279263953 0.08479021167198833 -0.212989282959229 -0.39546619238680936 0.5577269946072854 -0.5802114865911353 -0.3779644730092287]

η_Yb171(f) = 0.1924385393508112 / sqrt(f)
const ion1 = 2
const ion2 = 6

const modes = Seq.Modes()
for i in 1:7
    f = fs[i]
    push!(modes, 2π * f, bij[ion1] * bij[ion2] * η_Yb171(f))
end
@show modes

import ForwardDiff
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

function _objfunc(vals)
    dis = vals[1]
    disδ = vals[2]
    area = vals[3]
    areaδ = vals[4]
    τ = vals[5] / 30

    t1 = (dis + disδ + 1e-10)
    # t1 = (dis + disδ + areaδ^2 + 1e-10)
    t2 = (τ + 1)
    t3 = area^2

    return t1 * t2 / t3
end
const objfunc = autodiff(_objfunc)

# function objfunc(vals, grads)
#     dis = vals[1]
#     disδ = vals[2]
#     area = vals[3]
#     areaδ = vals[4]
#     τ = vals[5]

#     t1 = (dis + disδ + areaδ^2 + 1e-10)
#     t2 = (τ + 100)
#     t3 = area^2

#     res = t1 * t2 / t3
#     grads[1] = t2 / t3
#     grads[2] = t2 / t3
#     grads[3] = -2 * res / area
#     grads[4] = 2 * areaδ * t2 / t3
#     grads[5] = t1 / t3

#     return res
# end

const nlmodel = Seq.Objective(SL.pmask_tfm,
                              ((:dis2, 0), (:disδ2, 0), (:area, 0),
                               (:areaδ, 0), (:τ, 0)),
                              objfunc, modes, buf,
                              freq=Seq.FreqSpec(true, sym=false))
const nargs = Seq.nparams(nlmodel)
const tracker = Opts.NLVarTracker(nargs)
Opts.set_bound!(tracker, nlmodel.param.τ, 0.5, 3)
for Ω in nlmodel.param.Ωpoly
    Opts.set_bound!(tracker, Ω, 0.7, 0.7)
end
for ω in nlmodel.param.ωs
    Opts.set_bound!(tracker, ω, 2π * 2.2, 2π * 2.5)
end

# opt = NLopt.Opt(:LD_LBFGS, nargs) # 50 ms
opt = NLopt.Opt(:LD_SLSQP, nargs) # 13 ms
# opt = NLopt.Opt(:LD_VAR1, nargs) # 80 ms
# opt = NLopt.Opt(:LD_VAR2, nargs) # 50 ms
# opt = NLopt.Opt(:LD_TNEWTON_PRECOND_RESTART, nargs) # 148 ms
NLopt.xtol_rel!(opt, 1e-7)
NLopt.ftol_rel!(opt, 1e-7)
NLopt.min_objective!(opt, nlmodel)
NLopt.lower_bounds!(opt, Opts.lower_bounds(tracker))
NLopt.upper_bounds!(opt, Opts.upper_bounds(tracker))

@btime NLopt.optimize($opt, $(Opts.lower_bounds(tracker)))

best_obj = 1.0
best_params = nothing
@time for i in 1:1000
    global best_obj, best_params
    obj, params, ret = NLopt.optimize(opt, Opts.init_vars!(tracker))
    if getfield(NLopt, ret) < 0
        continue
    end
    if obj < best_obj
        best_obj = obj
        total_t = nlmodel(Val((:τ, 0)), params)
        dis = nlmodel(Val((:dis2, 0)), params)
        disδ = nlmodel(Val((:disδ2, 0)), params)
        area = nlmodel(Val((:area, 0)), params)
        areaδ = nlmodel(Val((:areaδ, 0)), params)
        @show best_obj, dis, disδ, area, areaδ, total_t
        best_params = params
    end
end
# @show best_params
# @show Seq.RawParams(nlmodel, best_params)
