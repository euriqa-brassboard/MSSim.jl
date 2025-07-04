#!/usr/bin/julia

push!(LOAD_PATH, joinpath(@__DIR__, "../../lib"))

using MSSim
using JuMP
using BenchmarkTools

const Opts = MSSim.Optimizers
const SS = MSSim.SegSeq
const SL = MSSim.SymLinear

# using Ipopt
# const model = Model(Ipopt.Optimizer)
# set_optimizer_attribute(model, "max_iter",
#                         parse(Int, get(ENV, "OPT_MAX_ITER", "30000")))
# # set_optimizer_attribute(model, "print_level", 5)
# set_optimizer_attribute(model, "print_level", 0)

using NLopt
const model = Model(NLopt.Optimizer)
# set_optimizer_attribute(model, "algorithm", :LN_COBYLA)
# set_optimizer_attribute(model, "algorithm", :LD_MMA)
set_optimizer_attribute(model, "algorithm", :LD_LBFGS)

nseg = 30

buf = SL.ComputeBuffer{nseg,Float64}(Val(SS.ValueMask(true, true, true, true, true, true)),
                                   Val(SS.ValueMask(true, true, true, true, true, true)));
kern = SL.Kernel(buf, Val(SL.ParamGradMask(true, true, true, true, true)));
args = Opts.gen_args(model, nseg, freq=Opts.FreqSpec(true))
Opts.register_kernel_funcs(model, kern)

modes = Opts.Modes()
push!(modes, 2.5)

dis = Opts.total_dis(model, args, modes)
cdis = Opts.total_cumdis(model, args, modes)
area = Opts.total_area(model, args, modes)
disδ = Opts.total_disδ(model, args, modes)
areaδ = Opts.total_areaδ(model, args, modes)
all_areaδ = Opts.all_areaδ(model, args, modes)

tracker = Opts.VarTracker()
push!(tracker, args.τ, 1, 10)
push!(tracker, args.Ωs.poly[1], 0.1, 1)
for ω in args.ωs
    push!(tracker, ω, 1.0, 4.0)
end
obj = @NLexpression(model, (dis + disδ + 1e-15) / abs(area)^2)
@NLobjective(model, Min, obj)

best_obj = 1.0
best_params = nothing
@time for _ in 1:1000
    global best_obj, best_params
    Opts.init_vars(tracker)
    JuMP.optimize!(model)
    if value(obj) < best_obj
        @show value(dis), value(disδ), value(cdis), value(area), value(areaδ), value(all_areaδ)
        best_obj = value(obj)
        best_params = values.(all_variables(model))
    end
end
