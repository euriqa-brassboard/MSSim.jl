#!/usr/bin/julia

module Optimizers

using ..SegSeq
using ..SymLinear

using JuMP
using StaticArrays

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
    elseif spec.mid_order == 0
        return orders, nodes[1:nseg], zeros(nseg)
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

# Parameter interface
function nparams end
function transform_argument end
function transform_gradient end

struct MSParams{NSeg,NAmp,Sym,FM,NAmpNode}
    # amp0::Vector{Float64}
    amps::NTuple{NAmp,MVector{NAmpNode,Float64}}
    τ::Int
    Ωbase::Union{Int,Nothing}
    Ωpoly::Union{Vector{Int},Nothing}
    ωs::Vector{Int}
end

function MSParams{NSeg}(;freq=FreqSpec(), amp=AmpSpec()) where NSeg
    amp_vals = Vector{Float64}[]
    base = nothing
    if amp.cb !== nothing
        Ωbase = MVector{NSeg + 1,Float64}(undef)
        if amp.sym
            nmax = NSeg ÷ 2 + 1
            for i in 1:nmax
                Ωbase = amp.cb(i)
                Ωbase[i] = Ωbase
                Ωbase[NSeg + 1 - i] = Ωbase
            end
        else
            nmax = NSeg + 1
            for i in 1:nmax
                Ωbase[i] = amp.cb(i)
            end
        end
        push!(amp_vals, Ωbase)
        base = 2
    end
    poly = nothing
    if amp.mid_order >= 0
        xs = [(2 * i / (NSeg + 2) - 1) for i in 1:NSeg + 1]
        if amp.end_order > 0
            Ωend = (xs .+ 1).^amp.end_order .+ (1 .- xs).^amp.end_order
        else
            Ωend = ones(NSeg + 1)
        end
        step = amp.sym ? 2 : 1
        poly = Int[]
        for order in 0:step:amp.mid_order
            push!(amp_vals, MVector{NSeg + 1,Float64}(xs.^order .* Ωend))
            push!(poly, length(amp_vals) + 1)
        end
    else
        @assert amp.end_order <= 0
    end
    NAmp = length(amp_vals)
    NFreq = (freq.modulate ? (freq.sym ? (NSeg + 1) ÷ 2 : NSeg) : 1)
    return MSParams{NSeg,NAmp,freq.sym,freq.modulate,NSeg + 1}(
        (amp_vals...,), 1, base, poly, (1:NFreq) .+ (1 + NAmp))
end

function nparams(params::MSParams{NSeg,NAmp,Sym,FM}) where {NSeg,NAmp,Sym,FM}
    return 1 + NAmp + (FM ? (Sym ? (NSeg + 1) ÷ 2 : NSeg) : 1)
end

@inline function transform_argument(params::MSParams{NSeg,NAmp,Sym,FM},
                                    args_raw, args_user) where {NSeg,NAmp,Sym,FM}
    # τ, amps..., freqs...
    @assert length(args_user) == 1 + NAmp + (FM ? (Sym ? (NSeg + 1) ÷ 2 : NSeg) : 1)
    @assert length(args_raw) == NSeg * 5
    @inbounds τ = args_user[1]
    # @inbounds prev_Ω = params.amp0[1]
    prev_Ω = 0.0
    @inbounds for ai in 1:NAmp
        prev_Ω = muladd(params.amps[ai][1], args_user[ai + 1], prev_Ω)
    end
    φ = 0.0
    invτ = 1 / τ
    @inbounds for i in 1:NSeg
        # Ω = params.amp0[i + 1]
        Ω = 0.0
        for ai in 1:NAmp
            Ω = muladd(params.amps[ai][i + 1], args_user[ai + 1], Ω)
        end
        args_raw[i * 5 - 4] = τ
        args_raw[i * 5 - 3] = prev_Ω
        args_raw[i * 5 - 2] = (Ω - prev_Ω) * invτ
        args_raw[i * 5 - 1] = φ
        if !FM
            ω = args_user[NAmp + 2]
        elseif !Sym || i <= (NSeg + 1) ÷ 2
            ω = args_user[NAmp + 1 + i]
        else
            ω = args_user[NAmp + 1 + (NSeg + 1 - i)]
        end
        args_raw[i * 5] = ω
        φ = muladd(ω, τ, φ)
        prev_Ω = Ω
    end
    return τ * NSeg
end
@inline function transform_gradient(params::MSParams{NSeg,NAmp,Sym,FM},
                                    grads_user, grads_raw,
                                    args_raw, args_user) where {NSeg,NAmp,Sym,FM}
    @assert length(grads_raw) == NSeg * 5
    # τ, amps..., freqs...
    @assert length(grads_user) == 1 + NAmp + (FM ? (Sym ? (NSeg + 1) ÷ 2 : NSeg) : 1)

    @inbounds τ = args_user[1]
    invτ = 1 / τ

    gradτ = 0.0
    @inbounds for ai in 1:NAmp
        grads_user[ai + 1] = 0
    end
    if !FM
        grads_user[NAmp + 2] = 0
    end

    sumω = 0.0
    @inbounds for i in 1:NSeg
        grτ = grads_raw[i * 5 - 4]
        grΩ = grads_raw[i * 5 - 3]
        grΩ′ = grads_raw[i * 5 - 2]
        grφ = grads_raw[i * 5 - 1]
        grω = grads_raw[i * 5]

        Ω′ = args_raw[i * 5 - 2]
        ω = args_raw[i * 5]

        grΩ′_τ = grΩ′ * invτ

        gradτ += muladd(grΩ′_τ, -Ω′, muladd(grφ, sumω, grτ))

        for ai in 1:NAmp
            grads_user[ai + 1] = muladd(params.amps[ai][i], grΩ - grΩ′_τ,
                                        muladd(params.amps[ai][i + 1], grΩ′_τ,
                                               grads_user[ai + 1]))
        end

        if !FM
            grads_user[NAmp + 2] += grω
        elseif !Sym || i <= (NSeg + 1) ÷ 2
            grads_user[NAmp + 1 + i] = grω
        else
            grads_user[NAmp + 1 + (NSeg + 1 - i)] += grω
        end

        sumω += ω
    end
    @inbounds grads_user[1] = gradτ
    sumgrφ = 0.0
    @inbounds for i in NSeg:-1:1
        if !FM
            idx = NAmp + 2
        elseif !Sym || i <= (NSeg + 1) ÷ 2
            idx = NAmp + 1 + i
        else
            idx = NAmp + 1 + (NSeg + 1 - i)
        end
        grads_user[idx] = muladd(sumgrφ, τ, grads_user[idx])
        sumgrφ += grads_raw[i * 5 - 1]
    end
    return
end

struct ModeInfo{Kern}
    kern::Kern
    ωm::Float64
    weight::Float64
end

struct MSObjective{pmask,ObjArg,NSeg,Param,Obj,Modes<:Tuple,NArgs,NObjArgs} <: Function
    modes::Modes
    param::Param
    obj::Obj
    args::MVector{NArgs,Float64}
    grads::MVector{NArgs,Float64}
    objargs::MVector{NObjArgs,Float64}
    objgrads::MVector{NObjArgs,Float64}
end

nparams(m::MSObjective) = nparams(m.param)

function MSObjective(pmask, ObjArg, obj::Obj, modes::Modes,
                     buf::SymLinear.ComputeBuffer{NSeg};
                     freq=FreqSpec(), amp=AmpSpec()) where {NSeg,Obj}
    modes_array = []
    for (modei, (ωm, weight)) in enumerate(modes.modes)
        kern = SymLinear.Kernel(buf, Val(pmask))
        push!(modes_array, ModeInfo(kern, ωm, weight))
    end
    modes = (modes_array...,)
    param = MSParams{NSeg}(freq=freq, amp=amp)
    NArgs = NSeg * 5
    NObjArgs = length(ObjArg)
    return MSObjective{pmask,ObjArg,NSeg,typeof(param),Obj,typeof(modes),NArgs,
                       NObjArgs}(modes, param, obj, MVector{NArgs,Float64}(undef),
                                 MVector{NArgs,Float64}(undef),
                                 MVector{NObjArgs,Float64}(undef),
                                 MVector{NObjArgs,Float64}(undef))
end

function _opt_muladd(@nospecialize(ex))
    if !Meta.isexpr(ex, :call)
        return ex
    end
    ex.args[2:end] = _opt_muladd.(ex.args[2:end])
    if ex.args[1] === :(+)
        if length(ex.args) == 2
            return ex.args[2]
        end
        mul_args = []
        other_args = []
        for arg in ex.args[2:end]
            if Meta.isexpr(arg, :call) && length(arg.args) >= 3 && arg.args[1] === :(*)
                push!(mul_args, arg)
            else
                push!(other_args, arg)
            end
        end
        if isempty(mul_args)
            return ex
        end
        if isempty(other_args)
            v = pop!(mul_args)
        elseif length(other_args) == 1
            v = other_args[1]
        else
            v = :(+($(other_args...),))
        end
        for mul_arg in mul_args
            m1 = mul_arg.args[2]
            if length(mul_arg.args) > 3
                m2 = :(*($(mul_arg.args[3:end]...),))
            else
                m2 = mul_arg.args[3]
            end
            v = :(muladd($m1, $m2, $v))
        end
        return v
    end
    return ex
end

function _generate_nlobj(ObjArg, NSeg, Modes, obj_ex, grads_out_var)
    nmodes = length(Modes.parameters)

    # 1. Forward propagation of values
    # input parameter -> pulse parameter
    func_ex = quote
        args = m.args
        grads = m.grads
        objargs = m.objargs
        objgrads = m.objgrads
        total_τ = transform_argument(m.param, args, x)
    end
    mode_vars = []
    # pulse parameter -> pulse parameter for each mode
    # -> compute all values for each modes
    for i in 1:nmodes
        mode_var = (mode=gensym(:mode),
                    ωm=gensym(:ωm),
                    weight=gensym(:weight),
                    kern=gensym(:kern),
                    args=gensym(:kargs),
                    res=gensym(:sres),
                    resval=gensym(:resval),
                    resgrad=gensym(:resgrad))
        push!(mode_vars, mode_var)
        push!(func_ex.args, quote
                  $(mode_var.mode) = m.modes[$i]
                  $(mode_var.ωm) = $(mode_var.mode).ωm
                  $(mode_var.weight) = $(mode_var.mode).weight
                  $(mode_var.kern) = $(mode_var.mode).kern
                  $(mode_var.args) = $(mode_var.kern).args
                  φ = 0.0
                  @inbounds for j in 1:$NSeg
                      τ = args[j * 5 - 4]
                      $(mode_var.args)[j * 5 - 4] = τ
                      $(mode_var.args)[j * 5 - 3] = args[j * 5 - 3]
                      $(mode_var.args)[j * 5 - 2] = args[j * 5 - 2]
                      $(mode_var.args)[j * 5 - 1] = args[j * 5 - 1] - φ
                      δ = args[j * 5]
                      $(mode_var.args)[j * 5] = δ - $(mode_var.ωm)
                      φ = muladd($(mode_var.ωm), τ, φ)
                  end
                  SymLinear.force_update!($(mode_var.kern))
                  $(mode_var.res) = $(mode_var.kern).result
                  $(mode_var.resval) = $(mode_var.res).val
                  $(mode_var.resgrad) = $(mode_var.res).grad.values
              end)
    end
    var_map = Dict{Tuple{Symbol,Int},Symbol}()
    get_vars(name) = (get_var(name, i) for i in 1:nmodes)
    function get_var(name, idx)
        key = (name, idx)
        if haskey(var_map, key)
            return var_map[key]
        end
        vval = gensym("$(name)_$(idx)")
        var_map[key] = vval
        ex = if idx != 0
            mode_var = mode_vars[idx]
            if name === :rdis
                :(real($(mode_var.resval).dis))
            elseif name === :idis
                :(imag($(mode_var.resval).dis))
            elseif name === :dis2
                iv = get_var(:idis, idx)
                rv = get_var(:rdis, idx)
                :(muladd($iv, $iv, $rv^2))
            elseif name === :area
                :($(mode_var.resval).area)
            elseif name === :rdisδ
                :(real($(mode_var.resval).disδ))
            elseif name === :idisδ
                :(imag($(mode_var.resval).disδ))
            elseif name === :disδ2
                iv = get_var(:idisδ, idx)
                rv = get_var(:rdisδ, idx)
                :(muladd($iv, $iv, $rv^2))
            elseif name === :areaδ
                :($(mode_var.resval).areaδ)
            elseif name === :areaδ2
                :($(get_var(:areaδ, idx))^2)
            elseif name === :rcumdis
                :(real($(mode_var.resval).cumdis))
            elseif name === :icumdis
                :(imag($(mode_var.resval).cumdis))
            elseif name === :cumdis2
                iv = get_var(:icumdis, idx)
                rv = get_var(:rcumdis, idx)
                :(muladd($iv, $iv, $rv^2))
            end
        elseif name === :dis2
            :(+($(get_vars(:dis2)...),))
        elseif name === :cumdis2
            :(+($(get_vars(:cumdis2)...),))
        elseif name === :disδ2
            :(+($(get_vars(:disδ2)...),))
        elseif name === :area
            _opt_muladd(:(+($((:($(mode_vars[i].weight) * $v)
                               for (i, v) in enumerate(get_vars(:area)))...),)))
        elseif name === :areaδ
            _opt_muladd(:(+($((:($(mode_vars[i].weight) * $v)
                               for (i, v) in enumerate(get_vars(:areaδ)))...),)))
        elseif name === :areaδ2
            :(+($(get_vars(:areaδ2)...),))
        elseif name === :τ
            :total_τ
        end
        push!(func_ex.args, :($vval = $ex))
        return vval
    end
    # Compute cost function input parameters
    for (i, (name, idx)) in enumerate(ObjArg)
        push!(func_ex.args, :(@inbounds objargs[$i] = $(get_var(name, idx))))
    end
    # Compute cost function with gradients
    if grads_out_var === nothing
        push!(func_ex.args, quote
                  return $(obj_ex)(objargs, objgrads)
              end)
        return func_ex
    end
    push!(func_ex.args, quote
              resval = $(obj_ex)(objargs, objgrads)
              if isempty($grads_out_var)
                  return resval
              end
          end)

    # 2. Backward propagation of gradients
    # Load cost function gradients
    objgrad_vars = Symbol[]
    for i in 1:length(ObjArg)
        @gensym objgrad
        push!(objgrad_vars, objgrad)
        push!(func_ex.args, :(@inbounds $objgrad = objgrads[$i]))
    end

    # Now we need to compute the gradient wrt each of the basic parameters
    # of the cost function (i.e. [ri]dis, [ri]cumdis, [ri]disδ, area, areaδ for each mode)

    # First, breakdown to single mode arguments
    grad_map1 = Dict{Tuple{Symbol,Int},Vector{Any}}()
    τ_grad_var = nothing
    for (i, (name, idx)) in enumerate(ObjArg)
        objgrad_var = objgrad_vars[i]
        if idx != 0
            push!(get!(grad_map1, (name, idx), []), objgrad_var)
            continue
        end
        if name === :dis2 || name === :cumdis2 || name === :disδ2 || name === :areaδ2
            for idx in 1:nmodes
                push!(get!(grad_map1, (name, idx), []), objgrad_var)
            end
        elseif name === :τ
            @assert τ_grad_var === nothing
            τ_grad_var = objgrad_var
        else
            @assert name === :area || name === :areaδ
            for idx in 1:nmodes
                push!(get!(grad_map1, (name, idx), []),
                      :($objgrad_var * $(mode_vars[idx].weight)))
            end
        end
    end
    comb_terms(terms) = length(terms) == 1 ? terms[1] : _opt_muladd(:(+($(terms...),)))

    grad_map2 = Dict{Tuple{Symbol,Int},Vector{Any}}()
    # Then back propagate to the basic parameters (i.e. get rid of the *2 parameters)
    for ((name, idx), terms) in grad_map1
        @assert idx != 0
        old_term = comb_terms(terms)
        @gensym val
        if name === :dis2
            push!(get!(grad_map2, (:rdis, idx), []),
                  :(2 * $(var_map[(:rdis, idx)]) * $val))
            push!(get!(grad_map2, (:idis, idx), []),
                  :(2 * $(var_map[(:idis, idx)]) * $val))
        elseif name === :cumdis2
            push!(get!(grad_map2, (:rcumdis, idx), []),
                  :(2 * $(var_map[(:rcumdis, idx)]) * $val))
            push!(get!(grad_map2, (:icumdis, idx), []),
                  :(2 * $(var_map[(:icumdis, idx)]) * $val))
        elseif name === :disδ2
            push!(get!(grad_map2, (:rdisδ, idx), []),
                  :(2 * $(var_map[(:rdisδ, idx)]) * $val))
            push!(get!(grad_map2, (:idisδ, idx), []),
                  :(2 * $(var_map[(:idisδ, idx)]) * $val))
        elseif name === :areaδ2
            push!(get!(grad_map2, (:areaδ, idx), []),
                  :(2 * $(var_map[(:areaδ, idx)]) * $val))
        else
            append!(get!(grad_map2, (name, idx), []), terms)
            continue
        end
        push!(func_ex.args, :($val = $old_term))
    end

    # For each mode, back propagate the gradients to the pulse parameters,
    # including the adjustments for mode frequencies
    for i in 1:nmodes
        mode_var = mode_vars[i]
        get_term(name) = if haskey(grad_map2, (name, i))
            @gensym s
            push!(func_ex.args, :($s = $(comb_terms(grad_map2[(name, i)]))))
            return s
        end
        rdis_scale = get_term(:rdis)
        idis_scale = get_term(:idis)
        area_scale = get_term(:area)
        rcumdis_scale = get_term(:rcumdis)
        icumdis_scale = get_term(:icumdis)
        rdisδ_scale = get_term(:rdisδ)
        idisδ_scale = get_term(:idisδ)
        areaδ_scale = get_term(:areaδ)

        set_grad(v, offset) = if i == 1
            :(grads[j * 5 - $offset] = $v)
        else
            :(grads[j * 5 - $offset] += $v)
        end
        function comb_grad_terms(vin)
            terms = []
            rdis_scale !== nothing && push!(terms, :(real($(vin).dis) * $rdis_scale))
            idis_scale !== nothing && push!(terms, :(imag($(vin).dis) * $idis_scale))
            area_scale !== nothing && push!(terms, :($(vin).area * $area_scale))
            rcumdis_scale !== nothing &&
                push!(terms, :(real($(vin).cumdis) * $rcumdis_scale))
            icumdis_scale !== nothing &&
                push!(terms, :(imag($(vin).cumdis) * $icumdis_scale))
            rdisδ_scale !== nothing && push!(terms, :(real($(vin).disδ) * $rdisδ_scale))
            idisδ_scale !== nothing && push!(terms, :(imag($(vin).disδ) * $idisδ_scale))
            areaδ_scale !== nothing && push!(terms, :($(vin).areaδ * $areaδ_scale))
            if isempty(terms)
                return 0.0
            else
                return comb_terms(terms)
            end
        end
        push!(func_ex.args, quote
                  gradτ_cum = 0.0
                  @inbounds for j in $NSeg:-1:1
                      g0τ = $(mode_var.resgrad)[j * 5 - 4]
                      g0Ω = $(mode_var.resgrad)[j * 5 - 3]
                      g0Ω′ = $(mode_var.resgrad)[j * 5 - 2]
                      g0φ = $(mode_var.resgrad)[j * 5 - 1]
                      g0ω = $(mode_var.resgrad)[j * 5]
                      g1τ = $(comb_grad_terms(:g0τ)) - gradτ_cum
                      g1Ω = $(comb_grad_terms(:g0Ω))
                      g1Ω′ = $(comb_grad_terms(:g0Ω′))
                      g1φ = $(comb_grad_terms(:g0φ))
                      g1ω = $(comb_grad_terms(:g0ω))
                      $(set_grad(:g1τ, 4))
                      $(set_grad(:g1Ω, 3))
                      $(set_grad(:g1Ω′, 2))
                      $(set_grad(:g1φ, 1))
                      $(set_grad(:g1ω, 0))
                      gradτ_cum = muladd(g1φ, $(mode_var.ωm), gradτ_cum)
                  end
              end)
    end

    if τ_grad_var !== nothing
        push!(func_ex.args, quote
                  @inbounds for j in 1:$NSeg
                      grads[j * 5 - 4] += $τ_grad_var
                  end
              end)
    end

    push!(func_ex.args, quote
              transform_gradient(m.param, $grads_out_var, grads, args, x)
              return resval
          end)
    return func_ex
end

@generated function (m::MSObjective{pmask,ObjArg,NSeg,Param,Obj,Modes})(x, grads_out) where {pmask,ObjArg,NSeg,Param,Obj,Modes}
    return _generate_nlobj(ObjArg, NSeg, Modes, :(m.obj), :grads_out)
end

function get_args(m::MSObjective{pmask,ObjArg,NSeg}, x) where {pmask,ObjArg,NSeg}
    res = Vector{Float64}(undef, NSeg * 5)
    transform_argument(m.param, res, x)
    return res
end

function _dummy_obj(x, grad)
    return x[1]
end

@generated function (m::MSObjective{pmask,ObjArg,NSeg,Param,Obj,Modes})(::Val{ObjArg2}, x) where {pmask,ObjArg,NSeg,Param,Obj,Modes,ObjArg2}
    return _generate_nlobj((ObjArg2,), NSeg, Modes, :_dummy_obj, nothing)
end

end
