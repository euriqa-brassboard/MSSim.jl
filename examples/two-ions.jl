#!/usr/bin/julia

push!(LOAD_PATH, joinpath(@__DIR__, "../../lib"))

using MSSim
using JuMP
using BenchmarkTools

const Opts = MSSim.Optimizers

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

# const modes = [MSSim.SymLinear.Mode{Float64}(ω, 1, 1)
#                for ω in range(2.0, 2.5, length=31)]

# const opt = Opts.AvgDisOpt{Float64}(model, modes, 2, 0.2, 400)
# const opt_res = @btime Opts.optimize!(opt, fill(2.25, 200), min_ω=1, max_ω=3)
# @show opt_res
# @show opt.eval_count[]

const modes = [MSSim.SymLinear.Mode{Float64}(ω, 1, 1)
               for ω in range(2.0, 2.5, length=2)]
const opt = Opts.AvgDisOpt{Float64}(model, modes, 2, 0.2, 10)
opt_res = (fill(0.0, 5), 10.0)
@time for i in 1:10000
    global opt_res
    new_res = Opts.optimize!(opt, rand(5) .* 2 .+ 1, min_ω=1, max_ω=3)
    if new_res[2] < opt_res[2]
        @show new_res
        opt_res = new_res
    end
end
@show opt_res
@show opt.eval_count[]
