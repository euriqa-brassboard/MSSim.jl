#!/usr/bin/julia

push!(LOAD_PATH, joinpath(@__DIR__, "../../lib"))

using MSSim
using JuMP
using BenchmarkTools

const Opts = MSSim.Optimizers
const SS = MSSim.SegSeq
const SL = MSSim.SymLinear

using NLopt
const model = Model(NLopt.Optimizer)
set_optimizer_attribute(model, "algorithm", :LD_LBFGS)
# set_optimizer_attribute(model, "algorithm", :LD_SLSQP)

nseg = 20

buf = SL.ComputeBuffer{nseg,Float64}(Val(SS.ValueMask(true, true, true, false, true, false)),
                                   Val(SS.ValueMask(true, true, true, false, true, false)))
# buf = SL.ComputeBuffer{nseg,Float64}(Val(SS.ValueMask(true, true, true, true, true, true)),
#                                      Val(SS.ValueMask(true, true, true, true, true, true)))
kern = SL.Kernel(buf, Val(SL.ParamGradMask(true, true, true, true, true)));
args = Opts.gen_args(model, nseg,
                     freq=Opts.FreqSpec(true, sym=false),
                     amp=Opts.AmpSpec(sym=false, mid_order=40, end_order=2))
Opts.register_kernel_funcs(model, kern)

modes = Opts.Modes()
push!(modes, 2.5)

dis = Opts.total_dis(model, args, modes)
# cdis = Opts.total_cumdis(model, args, modes)
area = Opts.total_area(model, args, modes)
disδ = Opts.total_disδ(model, args, modes)
# areaδ = Opts.total_areaδ(model, args, modes)
# all_areaδ = Opts.all_areaδ(model, args, modes)

tracker = Opts.VarTracker()
push!(tracker, args.τ, 1, 2)
for Ω in args.Ωs.poly
    push!(tracker, Ω, -0.5, 0.5)
end
for ω in args.ωs
    push!(tracker, ω, 1.0, 4.0)
end
obj = @NLexpression(model, (dis + disδ + 1e-5) / (1 + 50 * abs(area)))
@NLobjective(model, Min, obj)

best_obj = 1.0
best_params = nothing
@time for i in 1:1000
    global best_obj, best_params
    Opts.init_vars(tracker)
    JuMP.optimize!(model)
    if value(obj) < best_obj
        best_obj = value(obj)
        @show best_obj, value(dis), value(disδ), value(area) #, value(all_areaδ)
        best_params = values.(all_variables(model))
    end
end
