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

function register_kernel_funcs(model, kern::SymLinear.Kernel{NSeg,T,SDV,SDG};
                               prefix="", suffix="") where {NSeg,T,SDV,SDG}
    maskv = SegSeq.value_mask(SDV)
    maskg = SegSeq.value_mask(SDG)

    if maskv.dis && maskg.dis
        register(model, Symbol("$(prefix)rdis$(suffix)"),
                 NSeg * 5, @inline((args...)->SymLinear.value_rdis(kern, args...)),
                 @inline((g, args...)->SymLinear.grad_rdis(g, kern, args...)),
                 autodiff=false)
        register(model, Symbol("$(prefix)idis$(suffix)"),
                 NSeg * 5, @inline((args...)->SymLinear.value_idis(kern, args...)),
                 @inline((g, args...)->SymLinear.grad_idis(g, kern, args...)),
                 autodiff=false)
    end
    if maskv.area && maskg.area
        register(model, Symbol("$(prefix)area$(suffix)"),
                 NSeg * 5, @inline((args...)->SymLinear.value_area(kern, args...)),
                 @inline((g, args...)->SymLinear.grad_area(g, kern, args...)),
                 autodiff=false)
    end
    if maskv.cumdis && maskg.cumdis
        register(model, Symbol("$(prefix)rcumdis$(suffix)"),
                 NSeg * 5, @inline((args...)->SymLinear.value_rcumdis(kern, args...)),
                 @inline((g, args...)->SymLinear.grad_rcumdis(g, kern, args...)),
                 autodiff=false)
        register(model, Symbol("$(prefix)icumdis$(suffix)"),
                 NSeg * 5, @inline((args...)->SymLinear.value_icumdis(kern, args...)),
                 @inline((g, args...)->SymLinear.grad_icumdis(g, kern, args...)),
                 autodiff=false)
    end
    if maskv.disδ && maskg.disδ
        register(model, Symbol("$(prefix)rdisδ$(suffix)"),
                 NSeg * 5, @inline((args...)->SymLinear.value_rdisδ(kern, args...)),
                 @inline((g, args...)->SymLinear.grad_rdisδ(g, kern, args...)),
                 autodiff=false)
        register(model, Symbol("$(prefix)idisδ$(suffix)"),
                 NSeg * 5, @inline((args...)->SymLinear.value_idisδ(kern, args...)),
                 @inline((g, args...)->SymLinear.grad_idisδ(g, kern, args...)),
                 autodiff=false)
    end
    if maskv.areaδ && maskg.areaδ
        register(model, Symbol("$(prefix)areaδ$(suffix)"),
                 NSeg * 5, @inline((args...)->SymLinear.value_areaδ(kern, args...)),
                 @inline((g, args...)->SymLinear.grad_areaδ(g, kern, args...)),
                 autodiff=false)
    end
end

struct AmpSpec{CB}
    cb::CB
    sym::Bool
    mid_order::Int
    end_order::Int
    function AmpSpec(; cb::CB=nothing, sym=true, mid_order=0, end_order=0) where CB
        if end_order >= 0
            mid_order = max(0, mid_order)
        end
        @assert cb !== nothing || mid_order >= 0
        return new{CB}(cb, sym, mid_order, end_order)
    end
end

function _gen_base_args(spec::AmpSpec, m::Model, nseg, τ)
    if spec.cb === nothing
        return nothing, nothing, nothing
    end
    Ωnodes_base = Vector{Float64}(undef, nseg + 1)
    if spec.sym
        nmax = nseg ÷ 2 + 1
        for i in 1:nmax
            Ωbase = spec.cb(i)
            Ωnodes_base[i] = Ωbase
            Ωnodes_base[nseg + 1 - i] = Ωbase
        end
    else
        nmax = nseg + 1
        for i in 1:nmax
            Ωnodes_base[i] = spec.cb(i)
        end
    end
    scale = @variable(m, base_name="Ωscale")
    Ωs_base = Ωnodes_base[1:nseg] .* scale
    Ω′_base = Vector{Any}(undef, nseg)
    for i in 1:nseg
        dΩ = Ωnodes_base[i + 1] - Ωnodes_base[i]
        Ω′_base[i] = dΩ == 0 ? 0.0 : (dΩ .* scale / τ)
    end
    return scale, Ωs_base, Ω′_base
end

function _gen_poly_args(spec::AmpSpec, m::Model, nseg, τ)
    if spec.mid_order < 0
        @assert spec.end_order <= 0
        return nothing, nothing, nothing
    end
    norders = spec.sym ? (spec.mid_order ÷ 2 + 1) : (spec.mid_order + 1)
    orders = @variable(m, [i=1:norders], base_name="Ωorder")
    xs = [(2 * i / (nseg + 2) - 1) for i in 1:nseg + 1]
    if spec.sym
        xs .= xs.^2 # Use only even orders for symmetric function
    end
    nodes = [evalpoly(x, orders) for x in xs]
    if spec.end_order > 0
        xs = [(2 * i / (nseg + 2) - 1) for i in 1:nseg + 1]
        nodes .*= (xs .+ 1).^spec.end_order .+ (1 .- xs).^spec.end_order
    end
    return orders, nodes[1:nseg], (nodes[2:end] .- nodes[1:nseg]) ./ τ
end

function _gen_args(spec::AmpSpec, m::Model, nseg, τ)
    base, baseΩs, baseΩ′s = _gen_base_args(spec, m, nseg, τ)
    poly, polyΩs, polyΩ′s = _gen_poly_args(spec, m, nseg, τ)
    if baseΩs === nothing
        @assert polyΩs !== nothing
        Ωs = polyΩs
        Ω′s = polyΩ′s
    elseif polyΩs === nothing
        Ωs = baseΩs
        Ω′s = baseΩ′s
    else
        Ωs = baseΩs .+ polyΩs
        Ω′s = baseΩ′s .+ polyΩ′s
    end
    return (base=base, poly=poly), Ωs, Ω′s
end

struct FreqSpec
    sym::Bool
    modulate::Bool
    function FreqSpec(modulate=false; sym=true)
        return new(sym, modulate)
    end
end

function _fm_getter(nseg, τ, ωs, ωm)
    φs = Vector{Any}(undef, nseg)
    δs = Vector{Any}(undef, nseg)
    φ = 0
    for i in 1:nseg
        δ = ωs[i] - ωm
        φs[i] = φ
        δs[i] = δ
        if φ === 0
            φ = δ * τ
        else
            φ += δ * τ
        end
    end
    return φs, δs
end

function _gen_args(spec::FreqSpec, m::Model, nseg, τ)
    if !spec.modulate
        ω = @variable(m, base_name="ω")
        return ω, function (ωm)
            δφ = (ω - ωm) * τ
            return [(i == 1 ? 0.0 : (δφ * (i - 1))) for i in 1:nseg], fill(δ, nseg)
        end
    elseif spec.sym
        ωs = @variable(m, [i=1:(nseg + 1) ÷ 2], base_name="ω")
        full_ωs = Vector{VariableRef}(undef, nseg)
        for i in 1:length(ωs)
            full_ωs[i] = ωs[i]
            full_ωs[nseg + 1 - i] = ωs[i]
        end
        return ωs, ωm->_fm_getter(nseg, τ, full_ωs, ωm)
    else
        ωs = @variable(m, [i=1:nseg], base_name="ω")
        return ωs, ωm->_fm_getter(nseg, τ, ωs, ωm)
    end
end

struct Args{T,A,F,CB}
    τ::T
    Ωs::A
    ωs::F
    getter::CB
end

function gen_args(m::Model, nseg; freq=FreqSpec(), amp=AmpSpec())
    τ = @variable(m, base_name="τ")
    Ω_params, Ωs, Ω′s = _gen_args(amp, m, nseg, τ)
    ω_params, getter = _gen_args(freq, m, nseg, τ)
    return Args(τ, Ω_params, ω_params,
                function (ωm)
                    args = Vector{Any}(undef, nseg * 5)
                    φs, δs = getter(ωm)
                    for i in 1:nseg
                        args[i * 5 - 4] = τ
                        args[i * 5 - 3] = Ωs[i]
                        args[i * 5 - 2] = Ω′s[i]
                        args[i * 5 - 1] = φs[i]
                        args[i * 5] = δs[i]
                    end
                    return args
                end)
end

struct VarTracker
    vars::Vector{Tuple{VariableRef,Float64,Float64}}
    VarTracker() = new(Tuple{VariableRef,Float64,Float64}[])
end

function Base.push!(tracker::VarTracker, var::VariableRef, lb, ub)
    set_lower_bound(var, lb)
    set_upper_bound(var, ub)
    push!(tracker.vars, (var, lb, ub))
    return
end

function init_vars(tracker::VarTracker)
    for (var, lb, ub) in tracker.vars
        set_start_value(var, lb + (ub - lb) * rand())
    end
end

struct Modes
    modes::Vector{Tuple{Float64,Float64}}
    Modes() = new(Tuple{Float64,Float64}[])
end

function Base.push!(modes::Modes, ω, η=1.0)
    push!(modes.modes, (ω, η))
    return
end

get_args(args::Args, modes::Modes) = [args.getter(ω) for (ω, η) in modes.modes]

function total_dis(m::Model, args::Args, modes::Modes; prefix="", suffix="")
    r_f = Symbol("$(prefix)rdis$(suffix)")
    i_f = Symbol("$(prefix)idis$(suffix)")
    res = 0
    for kargs in get_args(args, modes)
        r_ex = :($r_f($kargs...))
        i_ex = :($i_f($kargs...))
        res = @NLexpression(m, res + r_ex^2 + i_ex^2)
    end
    return res
end

function total_cumdis(m::Model, args::Args, modes::Modes; prefix="", suffix="")
    r_f = Symbol("$(prefix)rcumdis$(suffix)")
    i_f = Symbol("$(prefix)icumdis$(suffix)")
    res = 0
    for kargs in get_args(args, modes)
        r_ex = :($r_f($kargs...))
        i_ex = :($i_f($kargs...))
        res = @NLexpression(m, res + r_ex^2 + i_ex^2)
    end
    return res
end

function total_area(m::Model, args::Args, modes::Modes; prefix="", suffix="")
    f = Symbol("$(prefix)area$(suffix)")
    res = 0
    for (kargs, (ω, η)) in zip(get_args(args, modes), modes.modes)
        ex = :($f($kargs...))
        res = @NLexpression(m, res + ex * η^2)
    end
    return res
end

function total_disδ(m::Model, args::Args, modes::Modes; prefix="", suffix="")
    r_f = Symbol("$(prefix)rdisδ$(suffix)")
    i_f = Symbol("$(prefix)idisδ$(suffix)")
    res = 0
    for kargs in get_args(args, modes)
        r_ex = :($r_f($kargs...))
        i_ex = :($i_f($kargs...))
        res = @NLexpression(m, res + r_ex^2 + i_ex^2)
    end
    return res
end

function total_areaδ(m::Model, args::Args, modes::Modes; prefix="", suffix="")
    f = Symbol("$(prefix)areaδ$(suffix)")
    res = 0
    for (kargs, (ω, η)) in zip(get_args(args, modes), modes.modes)
        ex = :($f($kargs...))
        res = @NLexpression(m, res + ex * η^2)
    end
    return res
end

function all_areaδ(m::Model, args::Args, modes::Modes; prefix="", suffix="")
    f = Symbol("$(prefix)areaδ$(suffix)")
    res = 0
    for (kargs, (ω, η)) in zip(get_args(args, modes), modes.modes)
        ex = :($f($kargs...))
        res = @NLexpression(m, res + ex^2)
    end
    return res
end

end
