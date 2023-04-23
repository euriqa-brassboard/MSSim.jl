#!/usr/bin/julia

push!(LOAD_PATH, joinpath(@__DIR__, "../../lib"))

using MSSim
using JuMP
using NLopt
using Ipopt
using BenchmarkTools

mutable struct OptResults{T}
    obj::T
    const grad::Vector{T}
    function OptResults{T}() where {T}
        return new(zero(T), T[])
    end
end

struct OptContext{T,Sys}
    sys::Sys
    res::OptResults{T}
    function OptContext{T}(modes::AbstractVector, τ, Ω, nseg) where {T}
        sys = MSSim.SymLinear.System{T}(modes, Val(true), Val(false), Val(true))
        resize!(sys.pulses, nseg)
        sys.pulses .= Ref(MSSim.SymLinear.Pulse{T}(τ, 0, 0, 0, 0))
        sys.pulses[1] = MSSim.SymLinear.Pulse{T}(τ, Ω, 0, 0, 0)
        res = OptResults{T}()
        resize!(res.grad, (nseg + 1) ÷ 2)
        return new{T,typeof(sys)}(sys, res)
    end
end

const eval_count = Ref(0)
function update!(opt::OptContext{T}, ωs) where T
    pulses = opt.sys.pulses
    nseg = length(pulses)
    nω = length(ωs)
    @assert nω == (nseg + 1) ÷ 2
    changed = false
    @inbounds for i in 1:nseg
        pulse = pulses[i]
        if i <= nω
            ω = ωs[i]
        else
            ω = ωs[nseg - i + 1]
        end
        if pulse.ω == ω
            continue
        end
        changed = true
        pulses[i] = MSSim.SymLinear.Pulse{T}(pulse.τ, pulse.dΩ,
                                             pulse.Ω′, pulse.dφ, ω)
    end
    if !changed
        return false
    end
    eval_count[] += 1
    MSSim.SymLinear.compute!(opt.sys)
    obj = zero(T)
    grad = opt.res.grad
    grad .= zero(T)
    cumdis = opt.sys.result.cumdis
    cumdis_grad = opt.sys.result.cumdis_grad
    nmode = length(opt.sys.modes)
    @inline @inbounds for mi in 1:nmode
        # Objective is sum of abs2(cumdis) for each mode
        cd = cumdis[mi]
        obj += abs2(cd)
        for si in 1:nseg
            cd_g = cumdis_grad[mi][si][end] # Gradient w.r.t. ω
            if si <= nω
                gi = si
            else
                gi = nseg - si + 1
            end
            grad[gi] = muladd(2, muladd(real(cd), real(cd_g), imag(cd) * imag(cd_g)),
                              grad[gi])
        end
    end
    opt.res.obj = obj
    return true
end

function objective_function(opt::OptContext, ωs...)
    try
        update!(opt, ωs)
    catch
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
        end
        rethrow()
    end
    return opt.res.obj
end

function gradient_function(g, opt::OptContext, ωs...)
    try
        update!(opt, ωs)
    catch
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
        end
        rethrow()
    end
    g .= opt.res.grad
    return
end

function optimize!(opt::OptContext, init_ωs)
    # model = Model(Ipopt.Optimizer)
    # set_optimizer_attribute(model, "max_iter",
    #                         parse(Int, get(ENV, "OPT_MAX_ITER", "30000")))
    # set_optimizer_attribute(model, "print_level", 5)
    # # set_optimizer_attribute(model, "print_level", 0)
    model = Model(NLopt.Optimizer)
    # set_optimizer_attribute(model, "algorithm", :LN_COBYLA)
    # set_optimizer_attribute(model, "algorithm", :LD_MMA)
    set_optimizer_attribute(model, "algorithm", :LD_LBFGS)

    nseg = length(opt.sys.pulses)
    nω = (nseg + 1) ÷ 2

    register(model, :f, nω,
             (ωs...)->objective_function(opt, ωs...),
             (g, ωs...)->gradient_function(g, opt, ωs...),
             autodiff=false)
    @variable(model, 1 .<= ωs[i=1:nω] .<= 3, start=init_ωs[i])
    @NLobjective(model, Min, f(ωs...))
    eval_count[] = 0
    @time JuMP.optimize!(model)
    # @btime JuMP.optimize!($model)
    @show eval_count[]
    ωsv = value.(ωs)
    objv = objective_function(opt, ωsv...)
    return ωsv, objv, opt.sys.result.area
end

# const modes = [MSSim.SymLinear.Mode{Float64}(2.1, 1, 1),
#                MSSim.SymLinear.Mode{Float64}(2.2, 1, 1),
#                MSSim.SymLinear.Mode{Float64}(2.3, 1, 1),
#                MSSim.SymLinear.Mode{Float64}(2.4, 1, 1),
#                MSSim.SymLinear.Mode{Float64}(2.5, 1, 1)]
const modes = [MSSim.SymLinear.Mode{Float64}(ω, 1, 1)
               for ω in range(2.0, 2.5, length=29 * 3)]

const opt = OptContext{Float64}(modes, 2, 0.2, 400 * 4)
# const opt_res = @btime optimize!(opt, fill(2.3, 200 * 4))
const opt_res = optimize!(opt, fill(2.3, 200 * 4))
@show opt_res
