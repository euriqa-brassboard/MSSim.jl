#!/usr/bin/julia

module Optimizers

using ..SegSeq
using ..SymLinear
import ..Sequence as Seq

import ..Sequence: total_dis, total_cumdis, total_area, total_disδ, total_areaδ, all_areaδ

using JuMP

function register_kernel_funcs(model, kern::SymLinear.Kernel{NSeg,T,SDV,SDG};
                               prefix="", suffix="") where {NSeg,T,SDV,SDG}
    maskv = SegSeq.value_mask(SDV)
    maskg = SegSeq.value_mask(SDG)

    function reg_func(fname)
        name = Symbol("$(prefix)$(fname)$(suffix)")
        op = add_nonlinear_operator(model, NSeg * 5, SymLinear.ValCb{fname}(kern),
                                    SymLinear.GradCb{fname}(kern), name=name)
        model[name] = op
    end

    if maskv.dis && maskg.dis
        reg_func(:rdis)
        reg_func(:idis)
        reg_func(:dis2)
    end
    if maskv.area && maskg.area
        reg_func(:area)
    end
    if maskv.cumdis && maskg.cumdis
        reg_func(:rcumdis)
        reg_func(:icumdis)
        reg_func(:cumdis2)
    end
    if maskv.disδ && maskg.disδ
        reg_func(:rdisδ)
        reg_func(:idisδ)
        reg_func(:disδ2)
    end
    if maskv.areaδ && maskg.areaδ
        reg_func(:areaδ)
    end
end

function _gen_base_args(spec::Seq.AmpSpec, m::Model, nseg, τ)
    if spec.cb === nothing
        return nothing, nothing, nothing
    end
    Ωnodes_base = Vector{Float64}(undef, nseg + 1)
    if spec.sym
        nmax = nseg ÷ 2 + 1
        for i in 1:nmax
            Ωbase = spec.cb((i - 1) / (nseg / 2) - 1)
            Ωnodes_base[i] = Ωbase
            Ωnodes_base[nseg + 1 - i] = Ωbase
        end
    else
        nmax = nseg + 1
        for i in 1:nmax
            Ωnodes_base[i] = spec.cb((i - 1) / (nseg / 2) - 1)
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

function _gen_poly_args(spec::Seq.AmpSpec, m::Model, nseg, τ)
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
    elseif spec.mid_order == 0
        return orders, nodes[1:nseg], zeros(nseg)
    end
    return orders, nodes[1:nseg], (nodes[2:end] .- nodes[1:nseg]) ./ τ
end

function _gen_args(spec::Seq.AmpSpec, m::Model, nseg, τ)
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

function _gen_args(spec::Seq.FreqSpec, m::Model, nseg, τ)
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

mutable struct ModeData
    const ωm::Float64
    const weight::Float64
    const args::Vector{Any}
    const kern::SymLinear.Kernel
    const suffix::String
    rdis
    idis
    dis2
    area
    rcumdis
    icumdis
    cumdis2
    rdisδ
    idisδ
    disδ2
    areaδ
    ModeData(ωm, weight, args, kern, suffix::String) =
        new(ωm, weight, args, kern, suffix)
end
function _get_val(mode_data::ModeData, m::Model, name::Symbol)
    if !isdefined(mode_data, name)
        setfield!(mode_data, name,
                  m[Symbol("$name$(mode_data.suffix)")](mode_data.args...))
    end
    return getfield(mode_data, name)
end

struct MSModel{pmask,T,A,F,CB,Buf}
    m::Model
    τ::T
    Ωs::A
    ωs::F
    getter::CB
    buf::Buf
    mode_data::Vector{ModeData}
    MSModel{pmask}(m::Model, τ::T, Ωs::A, ωs::F, getter::CB, buf::Buf,
                   mode_data::Vector{ModeData}) where {pmask,T,A,F,CB,Buf} =
        new{pmask,T,A,F,CB,Buf}(m, τ, Ωs, ωs, getter, buf, mode_data)
end

function MSModel{pmask}(m::Model, modes::Seq.Modes, buf::SymLinear.ComputeBuffer{NSeg};
                        freq=Seq.FreqSpec(), amp=Seq.AmpSpec()) where {NSeg,pmask}
    τ = @variable(m, base_name="τ")
    Ω_params, Ωs, Ω′s = _gen_args(amp, m, NSeg, τ)
    ω_params, ω_getter = _gen_args(freq, m, NSeg, τ)
    function param_getter(ωm)
        args = Vector{Any}(undef, NSeg * 5)
        φs, δs = ω_getter(ωm)
        for i in 1:NSeg
            args[i * 5 - 4] = τ
            args[i * 5 - 3] = Ωs[i]
            args[i * 5 - 2] = Ω′s[i]
            args[i * 5 - 1] = φs[i]
            args[i * 5] = δs[i]
        end
        return args
    end
    mode_data = ModeData[]
    for (modei, (ωm, weight)) in enumerate(modes.modes)
        kern = SymLinear.Kernel(buf, Val(pmask))
        suffix = "_m$modei"
        register_kernel_funcs(m, kern, suffix=suffix)
        push!(mode_data, ModeData(ωm, weight, param_getter(ωm), kern, suffix))
    end
    return MSModel{pmask}(m, τ, Ω_params, ω_params, param_getter, buf, mode_data)
end

JuMP.value(args::MSModel) = Seq.RawParams(value.(args.getter(0.0)))

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

for var in [:rdis, :idis, :dis2, :area, :rcumdis, :icumdis, :cumdis2,
              :rdisδ, :idisδ, :disδ2, :areaδ]
    @eval function $(Symbol("get_$var"))(model::MSModel, i)
        return _get_val(model.mode_data[i], model.m, $(QuoteNode(var)))
    end
end
nmodes(model::MSModel) = length(model.mode_data)

total_dis(model::MSModel) = sum(get_dis2(model, i) for i in 1:nmodes(model))
total_cumdis(model::MSModel) = sum(get_cumdis2(model, i) for i in 1:nmodes(model))
total_area(model::MSModel) = sum(get_area(model, i) * model.mode_data[i].weight
                                 for i in 1:nmodes(model))
total_disδ(model::MSModel) = sum(get_disδ2(model, i) for i in 1:nmodes(model))
total_areaδ(model::MSModel) = sum(get_areaδ(model, i) * model.mode_data[i].weight
                                   for i in 1:nmodes(model))
all_areaδ(model::MSModel) = sum(get_areaδ(model, i)^2 for i in 1:nmodes(model))

struct NLVarTracker
    vars::Vector{Tuple{Float64,Float64}}
    NLVarTracker(nargs) = new(fill((-Inf, Inf), nargs))
end

function set_bound!(tracker::NLVarTracker, idx, lb, ub)
    tracker.vars[idx] = (lb, ub)
    return
end

lower_bounds(tracker::NLVarTracker) = [lb for (lb, ub) in tracker.vars]
upper_bounds(tracker::NLVarTracker) = [ub for (lb, ub) in tracker.vars]

function init_vars!(tracker::NLVarTracker, vars=nothing)
    nvars = length(tracker.vars)
    if vars === nothing
        vars = Vector{Float64}(undef, nvars)
    end
    for vi in 1:nvars
        lb, ub = tracker.vars[vi]
        if isfinite(lb)
            if isfinite(ub)
                vars[vi] = lb + (ub - lb) * rand()
            else
                vars[vi] = lb + rand()
            end
        elseif isfinite(ub)
            vars[vi] = ub - rand()
        else
            vars[vi] = rand()
        end
    end
    return vars
end

end
