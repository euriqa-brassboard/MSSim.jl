#!/usr/bin/julia

push!(LOAD_PATH, joinpath(@__DIR__, "../../lib"))

using MSSim
using JuMP
using BenchmarkTools

const Opts = MSSim.Optimizers
const SS = MSSim.SegSeq
const SL = MSSim.SymLinear

using NLopt

# https://github.com/jump-dev/NLopt.jl/pull/262
Opts.check_nlopt(NLopt.Optimizer)

const model = Model(NLopt.Optimizer)
set_optimizer_attribute(model, "algorithm", :LD_LBFGS)
# set_optimizer_attribute(model, "algorithm", :LD_SLSQP)

nseg = 30

buf = SL.ComputeBuffer{nseg,Float64}(Val(SS.mask_allδ), Val(SS.mask_allδ))
# buf = SL.ComputeBuffer{nseg,Float64}(Val(SS.mask_full), Val(SS.mask_full))
# kern = SL.Kernel(buf, Val(SL.pmask_tfm))
# kern = SL.Kernel(buf, Val(SL.pmask_full))

modes = Opts.Modes()
for i in 1:5
    push!(modes, (2.1 + 0.1 * i) * 2π, (-1)^i)
end

msmod = Opts.MSModel{SL.pmask_tfm}(model, modes, buf,
                                   freq=Opts.FreqSpec(true, sym=false))

dis = Opts.total_dis(msmod)
# cdis = Opts.total_cumdis(msmod)
area = Opts.total_area(msmod)
disδ = Opts.total_disδ(msmod)
areaδ = Opts.total_areaδ(msmod)
all_areaδ = Opts.all_areaδ(msmod)

tracker = Opts.VarTracker()
push!(tracker, msmod.τ, 1, 6)
for Ω in msmod.Ωs.poly
    fix(Ω, 0.5)
    # push!(tracker, Ω, 0.1, 0.5)
end
for ω in msmod.ωs
    push!(tracker, ω, 2π * 2.0, 2π * 3.0)
end
obj = (dis + disδ + areaδ^2 + 1e-10) / area^2 * (msmod.τ + 2)
# obj = @NLexpression(model, (10 * dis + disδ + 1e-10) / (1 + abs(area)))
# obj = @NLexpression(model, (cdis + 1e-10) / (1 + abs(area)))
# obj = @NLexpression(model, cdis + 1e-10)
# obj = @NLexpression(model, dis + disδ)
@objective(model, Min, obj)

using BenchmarkTools
@btime JuMP.optimize!(model)

best_obj = 1.0
best_params = nothing
@time for i in 1:100
    global best_obj, best_params
    Opts.init_vars(tracker)
    @time JuMP.optimize!(model)
    if value(obj) < best_obj
        best_obj = value(obj)
        @show best_obj, value(dis), value(disδ), value(area), value(areaδ)
        best_params = value(msmod)
        @show best_params
    end
end
