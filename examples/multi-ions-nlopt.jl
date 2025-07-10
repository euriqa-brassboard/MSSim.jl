#!/usr/bin/julia

push!(LOAD_PATH, joinpath(@__DIR__, "../../lib"))

using MSSim
using BenchmarkTools

const Opts = MSSim.Optimizers
const SS = MSSim.SegSeq
const SL = MSSim.SymLinear

using NLopt

nseg = 30
buf = SL.ComputeBuffer{nseg,Float64}(Val(Opts.mask_allδ), Val(Opts.mask_allδ))
# buf = SL.ComputeBuffer{nseg,Float64}(Val(Opts.mask_full), Val(Opts.mask_full))
modes = Opts.Modes()
for i in 1:5
    push!(modes, (2.1 + 0.1 * i) * 2π, (-1)^i)
end

# import ForwardDiff
# function autodiff(f::F) where F
#     function fn_with_diff(x, grad)
#         if !isempty(grad)
#             # Use ForwardDiff to compute the gradient. Replace with your
#             # favorite Julia automatic differentiation package.
#             ForwardDiff.gradient!(grad, f, x)
#         end
#         return f(x)
#     end
# end

# function _objfunc(vals)
#     dis = vals[1]
#     disδ = vals[2]
#     area = vals[3]
#     areaδ = vals[4]
#     τ = vals[5] / 30

#     t1 = (dis + disδ + areaδ^2 + 1e-10)
#     t2 = (τ + 2)
#     t3 = area^2

#     return t1 * t2 / t3
# end
# const objfunc = autodiff(_objfunc)

function objfunc(vals, grads)
    dis = vals[1]
    disδ = vals[2]
    area = vals[3]
    areaδ = vals[4]
    τ = vals[5]

    t1 = (dis + disδ + areaδ^2 + 1e-10)
    t2 = (τ + 100)
    t3 = area^2

    res = t1 * t2 / t3
    grads[1] = t2 / t3
    grads[2] = t2 / t3
    grads[3] = -2 * res / area
    grads[4] = 2 * areaδ * t2 / t3
    grads[5] = t1 / t3

    return res
end

const nlmodel = Opts.MSObjective(Opts.pmask_tfm,
                                 ((:dis2, 0), (:disδ2, 0), (:area, 0),
                                  (:areaδ, 0), (:τ, 0)),
                                 objfunc, modes, buf,
                                 freq=Opts.FreqSpec(true, sym=false))
const nargs = Opts.nparams(nlmodel)

opt = NLopt.Opt(:LD_LBFGS, nargs) # 50 ms
# opt = NLopt.Opt(:LD_SLSQP, nargs) # 13 ms
# opt = NLopt.Opt(:LD_VAR1, nargs) # 80 ms
# opt = NLopt.Opt(:LD_VAR2, nargs) # 50 ms
# opt = NLopt.Opt(:LD_TNEWTON_PRECOND_RESTART, nargs) # 148 ms
NLopt.lower_bounds!(opt, [1; 0.5; fill(2π * 2.0, nargs - 2)])
NLopt.upper_bounds!(opt, [6; 0.5; fill(2π * 3.0, nargs - 2)])
NLopt.xtol_rel!(opt, 1e-7)
NLopt.ftol_rel!(opt, 1e-7)
NLopt.min_objective!(opt, nlmodel)

args_init = [1; 0.5; fill(2π * 2.0, nargs - 2)]
@btime NLopt.optimize($opt, $args_init)

best_obj = 1.0
best_params = nothing
@time for i in 1:100
    global best_obj, best_params
    obj, params, ret = NLopt.optimize(opt, [rand() * 5 + 1; 0.5;
                                            [(rand() + 2) * 2π
                                             for _ in 1:nargs - 2]])
    if getfield(NLopt, ret) < 0
        continue
    end
    if obj < best_obj
        best_obj = obj
        dis = nlmodel(Val((:dis2, 0)), params)
        disδ = nlmodel(Val((:disδ2, 0)), params)
        area = nlmodel(Val((:area, 0)), params)
        areaδ = nlmodel(Val((:areaδ, 0)), params)
        @show best_obj, dis, disδ, area, areaδ
        best_params = params
    end
end
@show best_params
@show Opts.get_args(nlmodel, best_params)
