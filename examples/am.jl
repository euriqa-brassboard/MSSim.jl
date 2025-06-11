#!/usr/bin/julia

push!(LOAD_PATH, joinpath(@__DIR__, "../../lib"))

using MSSim
using JuMP
using BenchmarkTools

const SS = MSSim.SegSeq
const SL = MSSim.SymLinear
const Opts = MSSim.Optimizers

const mask_val = SS.ValueMask(true, true, true, false, true, true)
const mask_grad = SS.ValueMask(true, true, true, false, true, true)

const pmask_am = SL.ParamGradMask(false, true, true, false, false)

function init_am_only_pulses!(pulses::AbstractVector{P}, nseg, τ, ω) where P<:SL.Pulse
    resize!(pulses, nseg)
    @inbounds for i in 1:nseg
        pulses[i] = P(τ, 0, 0, 0, ω)
    end
    return
end

struct DisGradOpt{T,Sys}
    sys::Sys
    cache::Opts.ObjCache{T}
    eval_count::Base.RefValue{Int}
    model::Model
    Ωs::Vector{VariableRef}
    function DisGradOpt{T}(model::Model, modes::AbstractVector, τ, ω, nseg) where {T}
        sys = SL.System{T}(modes, Val(mask_val), Val(mask_grad), Val(pmask_am))
        init_am_only_pulses!(sys.pulses, nseg, τ, ω)

        @variable(model, Ωs[i=1:(nseg - 1)])

        opt = new{T,typeof(sys)}(sys, Opts.ObjCache{T}(nseg - 1), Ref(0), model, Ωs)

        register(model, :f, nseg - 1, (Ωs...)->objective_function(opt, Ωs...),
                 (g, Ωs...)->gradient_function(g, opt, Ωs...), autodiff=false)

        @NLobjective(model, Min, f(Ωs...))

        return opt
    end
end

# TODO
function update!(opt::DisGradOpt{T}, Ωs) where T
    pulses = opt.sys.pulses
    nseg = length(pulses)
    nΩ = length(Ωs)
    @assert nΩ == nseg - 1
    changed = false
    @inbounds for i in 1:nseg
        pulse = pulses[i]
        Ω1 = i == 1 ? 0.0 : Ωs[i - 1]
        Ω2 = i == nseg ? 0.0 : Ωs[i]
        Ω′ = (Ω2 - Ω1) / pulse.τ
        if pulse.Ω′ == Ω′
            continue
        end
        changed = true
        pulses[i] = SL.Pulse{T}(pulse.τ, pulse.dΩ, Ω′, pulse.dφ, pulse.ω)
    end
    if !changed
        return false
    end
    opt.eval_count[] += 1
    SL.compute!(opt.sys)

    dis = opt.sys.result.dis
    dis_grad = opt.sys.result.dis_grad
    disδ = opt.sys.result.disδ
    disδ_grad = opt.sys.result.disδ_grad
    area_grad = opt.sys.result.area_grad
    areaδ = opt.sys.result.areaδ
    areaδ_grad = opt.sys.result.areaδ_grad

    obj = zero(T) # (opt.sys.result.area - 1)^2
    grad = opt.cache.grad
    grad .= zero(T)
    # for si in 1:(nseg - 1)
    #     g1Ω′ = area_grad[si][end] # Gradient w.r.t. Ω′
    #     g2Ω′ = area_grad[si + 1][end] # Gradient w.r.t. Ω′
    #     grad[si] = 2 * (opt.sys.result.area - 1) * (g1Ω′ - g2Ω′) / pulses[si].τ
    # end

    nmode = length(opt.sys.modes)

    @inline @inbounds for mi in 1:nmode
        # Objective is sum of abs2(dis) + abs2(disδ) + abs2(areaδ) for each mode
        d = dis[mi]
        dδ = disδ[mi]
        aδ = areaδ[mi]
        obj += abs2(d) + abs2(dδ) + abs2(aδ)
        for si in 1:(nseg - 1)
            d1_g = dis_grad[mi][si][end] # Gradient w.r.t. Ω′
            d2_g = dis_grad[mi][si + 1][end] # Gradient w.r.t. Ω′
            dδ1_g = disδ_grad[mi][si][end] # Gradient w.r.t. Ω′
            dδ2_g = disδ_grad[mi][si + 1][end] # Gradient w.r.t. Ω′
            aδ1_g = areaδ_grad[mi][si][end] # Gradient w.r.t. Ω′
            aδ2_g = areaδ_grad[mi][si + 1][end] # Gradient w.r.t. Ω′
            d_g = (d1_g - d2_g) / pulses[si].τ
            dδ_g = (dδ1_g - dδ2_g) / pulses[si].τ
            aδ_g = (aδ1_g - aδ2_g) / pulses[si].τ
            grad[si] = muladd(2, muladd(real(d), real(d_g), imag(d) * imag(d_g)),
                              grad[si])
            grad[si] = muladd(2, muladd(real(dδ), real(dδ_g), imag(dδ) * imag(dδ_g)),
                              grad[si])
            grad[si] = muladd(2, aδ * aδ_g, grad[si])
        end
    end
    opt.cache.obj = obj
    return true
end

function objective_function(opt::DisGradOpt, Ωs...)
    try
        update!(opt, Ωs)
    catch
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
        end
        rethrow()
    end
    return opt.cache.obj
end

function gradient_function(g, opt::DisGradOpt, Ωs...)
    try
        update!(opt, Ωs)
        g .= opt.cache.grad
    catch
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
        end
        rethrow()
    end
    return
end

function optimize!(opt::DisGradOpt, init_Ωs; min_Ω=0, max_Ω=nothing)
    nseg = length(opt.sys.pulses)
    nΩ = nseg - 1

    @assert length(init_Ωs) == nΩ
    for i in 1:nΩ
        @inbounds Ωv = opt.Ωs[i]
        if i == (nΩ + 1) ÷ 2
            fix(Ωv, 1)
            continue
        end
        @inbounds set_start_value(Ωv, init_Ωs[i])
        if min_Ω === nothing
            delete_lower_bound(Ωv)
        else
            set_lower_bound(Ωv, min_Ω)
        end
        if max_Ω === nothing
            delete_upper_bound(Ωv)
        else
            set_upper_bound(Ωv, max_Ω)
        end
    end

    opt.eval_count[] = 0
    JuMP.optimize!(opt.model)
    Ωsv = value.(opt.Ωs)
    objv = objective_function(opt, Ωsv...)
    return Ωsv, objv, opt.sys.result.area
end

# 2 MHz harmonic axial confinement, 2 MHz radial confinement
const mode_freqs = [1.2267976851147497, 1.527602479537847, 1.7425443388437865, 1.897366596294101, 2.0]
const mode_bijs = [-0.10454003077778197 -0.30165793484611997 0.5376535891863655 -0.6395330252179114 -0.44721359549995776; 0.4704366784410061 0.6395330252179118 -0.28051618772789594 -0.3016579348461201 -0.4472135954999583; -0.7317932953264503 -1.517497407323049e-15 -0.5142748029169374 1.0283773695369006e-16 -0.44721359549995826; 0.47043667844100817 -0.6395330252179116 -0.2805161877278957 0.30165793484611997 -0.44721359549995765; -0.10454003077778247 0.3016579348461202 0.5376535891863644 0.6395330252179118 -0.4472135954999578]

using Ipopt
const model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "max_iter",
                        parse(Int, get(ENV, "OPT_MAX_ITER", "30000")))
set_optimizer_attribute(model, "print_level", 5)
# set_optimizer_attribute(model, "print_level", 0)

# using NLopt
# const model = Model(NLopt.Optimizer)
# # set_optimizer_attribute(model, "algorithm", :LN_COBYLA)
# # set_optimizer_attribute(model, "algorithm", :LD_MMA)
# set_optimizer_attribute(model, "algorithm", :LD_LBFGS)

const modes = [SL.Mode{Float64}(mode_freqs[i],
                                mode_bijs[1, i] + mode_bijs[2, i],
                                mode_bijs[1, i] * mode_bijs[2, i]) for i in 1:5]
const opt = DisGradOpt{Float64}(model, modes, 2, 1.6, 200)
const opt_res = optimize!(opt, fill(1, 199), min_Ω=0, max_Ω=200)
@show opt_res
@show opt.eval_count[]
