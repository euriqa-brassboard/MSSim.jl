#!/usr/bin/julia

module Optimizers

using ..SegSeq
using ..SymLinear

using JuMP

const mask_full = SegSeq.ValueMask(true, true, true, true, true, true)
const mask_allδ = SegSeq.ValueMask(true, true, true, false, true, true)

const pmask_full = SymLinear.ParamGradMask(true, true, true, true, true)
const pmask_fm = SymLinear.ParamGradMask(false, false, false, true, true)
const pmask_tfm = SymLinear.ParamGradMask(true, false, false, true, true)
const pmask_am = SymLinear.ParamGradMask(false, true, true, false, false)
const pmask_tam = SymLinear.ParamGradMask(true, true, true, false, false)

mutable struct ObjCache{T}
    obj::T
    const grad::Vector{T}
    function ObjCache{T}(nvars=0) where {T}
        return new(zero(T), Vector{T}(undef, nvars))
    end
end

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

struct Modes
    modes::Vector{Tuple{Float64,Float64}}
    Modes() = new(Tuple{Float64,Float64}[])
end

function Base.push!(modes::Modes, ω, weight=1.0)
    push!(modes.modes, (ω, weight))
    return
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

function MSModel{pmask}(m::Model, modes::Modes, buf::SymLinear.ComputeBuffer{NSeg};
                        freq=FreqSpec(), amp=AmpSpec()) where {NSeg,pmask}
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

struct ArgsValue
    args::Vector{Float64}
end

function adjust(args::ArgsValue; tmax=Inf)
    res = copy(args.args)
    nargs = length(res)
    @assert nargs % 5 == 0
    nseg = nargs ÷ 5
    totalt = 0.0
    for i in 1:nseg
        if totalt >= tmax
            res[i * 5 - 4] = 0
            continue
        end
        τ = res[i * 5 - 4]
        dt = tmax - totalt
        totalt += τ
        if dt < τ
            res[i * 5 - 4] = dt
        end
    end
    return ArgsValue(res)
end

JuMP.value(args::MSModel) = ArgsValue(value.(args.getter(0.0)))
function get_args(args::ArgsValue, ωm)
    res = copy(args.args)
    nargs = length(res)
    @assert nargs % 5 == 0
    nseg = nargs ÷ 5
    φ = 0.0
    for i in 1:nseg
        τ = res[i * 5 - 4]
        res[i * 5 - 1] -= φ
        φ += ωm * τ
        res[i * 5] -= ωm
    end
    return res
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

get_args(args::ArgsValue, modes::Modes; δ=0.0) =
    [get_args(args, ω + δ) for (ω, _) in modes.modes]

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

function total_dis(kern::SymLinear.Kernel, args::ArgsValue, modes::Modes; δ=0.0)
    res = 0.0
    for kargs in get_args(args, modes; δ=δ)
        res += SymLinear.value_dis2(kern, kargs...)
    end
    return res
end

function total_cumdis(kern::SymLinear.Kernel, args::ArgsValue, modes::Modes; δ=0.0)
    res = 0.0
    for kargs in get_args(args, modes; δ=δ)
        res += SymLinear.value_cumdis2(kern, kargs...)
    end
    return res
end

function total_area(kern::SymLinear.Kernel, args::ArgsValue, modes::Modes; δ=0.0)
    res = 0.0
    for (kargs, (ω, weight)) in zip(get_args(args, modes; δ=δ), modes.modes)
        res += SymLinear.value_area(kern, kargs...) * weight
    end
    return res
end

function total_disδ(kern::SymLinear.Kernel, args::ArgsValue, modes::Modes; δ=0.0)
    res = 0.0
    for kargs in get_args(args, modes; δ=δ)
        res += SymLinear.value_disδ2(kern, kargs...)
    end
    return res
end

function total_areaδ(kern::SymLinear.Kernel, args::ArgsValue, modes::Modes; δ=0.0)
    res = 0.0
    for (kargs, (ω, weight)) in zip(get_args(args, modes; δ=δ), modes.modes)
        res += SymLinear.value_areaδ(kern, kargs...) * weight
    end
    return res
end

function all_areaδ(kern::SymLinear.Kernel, args::ArgsValue, modes::Modes; δ=0.0)
    res = 0.0
    for kargs in get_args(args, modes; δ=δ)
        res += SymLinear.value_areaδ(kern, kargs...)^2
    end
    return res
end

# Temporary function to patch NLopt to support new non-linear function interface
import MathOptInterface as MOI
function check_nlopt(Optimizer)
    opt = Optimizer()
    attr = MOI.UserDefinedFunction(:dummy, 2)
    if MOI.supports(opt, attr)
        return
    end
    M = Optimizer.name.module
    @eval M begin
        MOI.supports(model::Optimizer, ::MOI.UserDefinedFunction) = true
        function MOI.set(model::Optimizer, attr::MOI.UserDefinedFunction, args)
            _init_nlp_model(model)
            MOI.Nonlinear.register_operator(
                model.nlp_model,
                attr.name,
                attr.arity,
                args...,
            )
            return
        end

        function MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators)
            _init_nlp_model(model)
            return MOI.get(model.nlp_model, attr)
        end
    end
end

end
