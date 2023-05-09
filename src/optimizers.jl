#!/usr/bin/julia

module Optimizers

using ..SegSeq
using ..SymLinear

using JuMP

function init_fm_only_pulses!(pulses::AbstractVector{P}, nseg, τ, Ω) where P<:SymLinear.Pulse
    resize!(pulses, nseg)
    pulses[1] = P(τ, Ω, 0, 0, 0)
    @inbounds for i in 2:nseg
        pulses[i] = P(τ, 0, 0, 0, 0)
    end
    return
end

const mask_full = SegSeq.ValueMask(true, true, true, true, true, true)
const mask_τ_avgdis = SegSeq.ValueMask(true, true, false, true, false, false)
const mask_τ_area_avgdis = SegSeq.ValueMask(true, true, true, true, false, false)
const mask_avgdis = SegSeq.ValueMask(false, true, false, true, false, false)

const pmask_full = SymLinear.ParamGradMask(true, true, true, true, true)
const pmask_δ = SymLinear.ParamGradMask(false, false, false, false, true)

mutable struct ObjCache{T}
    obj::T
    const grad::Vector{T}
    function ObjCache{T}(nvars=0) where {T}
        return new(zero(T), Vector{T}(undef, nvars))
    end
end


struct AvgDisOpt{T,Sys}
    sys::Sys
    cache::ObjCache{T}
    eval_count::Base.RefValue{Int}
    model::Model
    ωs::Vector{VariableRef}
    function AvgDisOpt{T}(model::Model, modes::AbstractVector, τ, Ω, nseg) where {T}
        sys = SymLinear.System{T}(modes, Val(mask_τ_avgdis),
                                  Val(mask_avgdis), Val(pmask_δ))
        init_fm_only_pulses!(sys.pulses, nseg, τ, Ω)
        nω = (nseg + 1) ÷ 2

        @variable(model, ωs[i=1:nω])

        opt = new{T,typeof(sys)}(sys, ObjCache{T}(nω), Ref(0), model, ωs)

        register(model, :f, nω, (ωs...)->objective_function(opt, ωs...),
                 (g, ωs...)->gradient_function(g, opt, ωs...), autodiff=false)

        @NLobjective(model, Min, f(ωs...))

        return opt
    end
end

function update!(opt::AvgDisOpt{T}, ωs) where T
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
        pulses[i] = SymLinear.Pulse{T}(pulse.τ, pulse.dΩ, pulse.Ω′, pulse.dφ, ω)
    end
    if !changed
        return false
    end
    opt.eval_count[] += 1
    SymLinear.compute!(opt.sys)
    obj = zero(T)
    grad = opt.cache.grad
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
    opt.cache.obj = obj
    return true
end

function objective_function(opt::AvgDisOpt, ωs...)
    try
        update!(opt, ωs)
    catch
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
        end
        rethrow()
    end
    return opt.cache.obj
end

function gradient_function(g, opt::AvgDisOpt, ωs...)
    try
        update!(opt, ωs)
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

function optimize!(opt::AvgDisOpt, init_ωs; min_ω=nothing, max_ω=nothing)
    nseg = length(opt.sys.pulses)
    nω = (nseg + 1) ÷ 2

    @assert length(init_ωs) == nω
    for i in 1:nω
        @inbounds ωv = opt.ωs[i]
        @inbounds set_start_value(ωv, init_ωs[i])
        if min_ω === nothing
            delete_lower_bound(ωv)
        else
            set_lower_bound(ωv, min_ω)
        end
        if max_ω === nothing
            delete_upper_bound(ωv)
        else
            set_upper_bound(ωv, max_ω)
        end
    end

    opt.eval_count[] = 0
    JuMP.optimize!(opt.model)
    ωsv = value.(opt.ωs)
    objv = objective_function(opt, ωsv...)
    return ωsv, objv
end

end
