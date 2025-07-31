#!/usr/bin/julia

module Sequence

import ..SegSeq as SS
import ..SymLinear as SL
using StaticArrays

struct AmpSpec{CBs<:Tuple}
    cbs::CBs
    sym::Bool
    function AmpSpec(; cb::CB=_->1.0, sym=true) where CB
        _CB = CB
        if !isa(cb, Tuple)
            cb = (cb,)
            _CB = Tuple{_CB}
        end
        return new{_CB}(cb, sym)
    end
end

struct FreqSpec
    sym::Bool
    modulate::Bool
    function FreqSpec(modulate=false; sym=true)
        return new(sym, modulate)
    end
end

# Sequence parameter interface
"""
    nparams(modspec) -> Int

Return number of free parameters
"""
function nparams end

"""
    transform_argument(modspec, args_raw, args_user) -> Float64

Convert user arguments to raw sequence parameters.
Return total sequence time.
"""
function transform_argument end

"""
    transform_gradient(modspec, grads_user, grads_raw, args_raw, args_user)

Convert gradients wrt. raw parameters to gradients wrt. user arguments.
The raw parameters and the user arguments are also provided.
"""
function transform_gradient end

struct ModSpec{NSeg,NAmp,Sym,FM,NAmpNode}
    amps::NTuple{NAmp,MVector{NAmpNode,Float64}}
    τ::Int
    Ωs::NTuple{NAmp,Int}
    ωs::Vector{Int}
end

function ModSpec{NSeg}(;freq=FreqSpec(), amp=AmpSpec()) where NSeg
    NAmp = length(amp.cbs)
    amps = ntuple(_->MVector{NSeg + 1,Float64}(undef), Val(NAmp))
    Ωs_idx = ntuple(i->i + 1, Val(NAmp))
    for (cb, Ωs) in zip(amp.cbs, amps)
        if amp.sym
            for i in 1:NSeg ÷ 2 + 1
                Ω = cb((i - 1) / (NSeg / 2) - 1)
                Ωs[i] = Ω
                Ωs[NSeg + 2 - i] = Ω
            end
        else
            for i in 1:NSeg + 1
                Ωs[i] = cb((i - 1) / (NSeg / 2) - 1)
            end
        end
    end
    NFreq = (freq.modulate ? (freq.sym ? (NSeg + 1) ÷ 2 : NSeg) : 1)
    return ModSpec{NSeg,NAmp,freq.sym,freq.modulate,NSeg + 1}(
        amps, 1, Ωs_idx, (1:NFreq) .+ (1 + NAmp))
end

function nparams(params::ModSpec{NSeg,NAmp,Sym,FM}) where {NSeg,NAmp,Sym,FM}
    return 1 + NAmp + (FM ? (Sym ? (NSeg + 1) ÷ 2 : NSeg) : 1)
end

@inline function transform_argument(params::ModSpec{NSeg,NAmp,Sym,FM},
                                    args_raw, args_user) where {NSeg,NAmp,Sym,FM}
    # τ, amps..., freqs...
    @assert length(args_user) == 1 + NAmp + (FM ? (Sym ? (NSeg + 1) ÷ 2 : NSeg) : 1)
    @assert length(args_raw) == NSeg * 5
    @inbounds τ = args_user[1]
    prev_Ω = 0.0
    @inbounds for ai in 1:NAmp
        prev_Ω = muladd(params.amps[ai][1], args_user[ai + 1], prev_Ω)
    end
    φ = 0.0
    invτ = 1 / τ
    @inbounds for i in 1:NSeg
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
@inline function transform_gradient(params::ModSpec{NSeg,NAmp,Sym,FM},
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

struct Modes
    modes::Vector{Tuple{Float64,Float64}}
    Modes() = new(Tuple{Float64,Float64}[])
end
Base.getindex(modes::Modes, i) = modes.modes[i]
function Base.push!(modes::Modes, ω, weight=1.0)
    push!(modes.modes, (ω, weight))
    return
end

struct RawParams
    args::Vector{Float64}
end

function gate_solution_info(params::RawParams; normalize_amp=true)
    nargs = length(params.args)
    @assert nargs % 5 == 0
    nseg = nargs ÷ 5

    if normalize_amp
        max_amp1 = maximum(abs.(@view params.args[2:5:end]))
        max_amp2 = maximum(abs.(@view(params.args[2:5:end])
                                .+ @view(params.args[1:5:end]) .*
                                    @view(params.args[3:5:end])))
        max_amp = max(max_amp1, max_amp2)
    else
        max_amp = 1.0
    end

    return Dict("time"=>params.args[1:5:end],
                "amp"=>params.args[2:5:end] ./ max_amp,
                "amp_slope"=>params.args[3:5:end] ./ max_amp,
                "phase"=>(params.args[4:5:end] ./ 2π) .% 1,
                "phase_slope"=>params.args[5:5:end] ./ 2π,
                "nsteps"=>nseg)
end

function Base.get(args::RawParams; tmax=Inf, δ=0.0, ωm=0.0)
    res = copy(args.args)
    nargs = length(res)
    @assert nargs % 5 == 0
    nseg = nargs ÷ 5

    δ -= ωm

    totalt = 0.0
    φ = 0.0
    for i in 1:nseg
        res[i * 5] += δ
        res[i * 5 - 1] += φ
        if totalt >= tmax
            res[i * 5 - 4] = 0
            continue
        end
        τ = res[i * 5 - 4]

        dt = tmax - totalt
        totalt += τ
        τ = min(dt, τ)
        φ += δ * τ
        res[i * 5 - 4] = τ
    end
    return res
end
_each_mode_arg(args::RawParams, modes::Modes; δ=0.0) =
    (get(args, ωm=ω, δ=δ) for (ω, _) in modes.modes)

adjust(args::RawParams; kws...) = RawParams(get(args; kws...))

function get_trajectory(params::RawParams, npoints; kws...)
    args = get(params; kws...)
    nargs = length(args)
    @assert nargs % 5 == 0
    nseg = nargs ÷ 5
    total_time = 0.0
    total_dis = 0.0im
    dis_offsets = ComplexF64[]
    for i in 1:nseg
        τ = args[i * 5 - 4]
        total_time += τ
        push!(dis_offsets, total_dis)
        total_dis += SL.SegInt.displacement(τ, args[i * 5 - 3], args[i * 5 - 2],
                                            args[i * 5 - 1], args[i * 5])
    end
    ts = range(0, total_time, npoints)
    xs = Vector{Float64}(undef, npoints)
    ys = Vector{Float64}(undef, npoints)
    ti = 1
    total_time = 0.0
    for i in 1:nseg
        time_offset = total_time
        τ = args[i * 5 - 4]
        total_time += τ
        dis_offset = dis_offsets[i]
        Ω = args[i * 5 - 3]
        Ω′ = args[i * 5 - 2]
        φ = args[i * 5 - 1]
        δ = args[i * 5]
        while ti <= npoints && ts[ti] <= total_time
            dis = dis_offset + SL.SegInt.displacement(ts[ti] - time_offset, Ω, Ω′, φ, δ)
            xs[ti] = real(dis)
            ys[ti] = imag(dis)
            ti += 1
        end
        if ti > npoints
            break
        end
    end
    xs[ti:end] .= real(total_dis)
    ys[ti:end] .= imag(total_dis)
    return ts, xs, ys
end

function total_dis(kern::SL.Kernel, args::RawParams, modes::Modes; δ=0.0)
    res = 0.0
    for kargs in _each_mode_arg(args, modes; δ=δ)
        res += SL.value_dis2(kern, kargs...)
    end
    return res
end

function total_cumdis(kern::SL.Kernel, args::RawParams, modes::Modes; δ=0.0)
    res = 0.0
    for kargs in _each_mode_arg(args, modes; δ=δ)
        res += SL.value_cumdis2(kern, kargs...)
    end
    return res
end

function total_area(kern::SL.Kernel, args::RawParams, modes::Modes; δ=0.0)
    res = 0.0
    for (kargs, (ω, weight)) in zip(_each_mode_arg(args, modes; δ=δ), modes.modes)
        res += SL.value_area(kern, kargs...) * weight
    end
    return res
end

function total_disδ(kern::SL.Kernel, args::RawParams, modes::Modes; δ=0.0)
    res = 0.0
    for kargs in _each_mode_arg(args, modes; δ=δ)
        res += SL.value_disδ2(kern, kargs...)
    end
    return res
end

function total_areaδ(kern::SL.Kernel, args::RawParams, modes::Modes; δ=0.0)
    res = 0.0
    for (kargs, (ω, weight)) in zip(_each_mode_arg(args, modes; δ=δ), modes.modes)
        res += SL.value_areaδ(kern, kargs...) * weight
    end
    return res
end

function all_areaδ(kern::SL.Kernel, args::RawParams, modes::Modes; δ=0.0)
    res = 0.0
    for kargs in _each_mode_arg(args, modes; δ=δ)
        res += SL.value_areaδ(kern, kargs...)^2
    end
    return res
end

function get_Ωs(args::RawParams)
    nargs = length(args.args)
    @assert nargs % 5 == 0
    nseg = nargs ÷ 5

    ts = Float64[]
    Ωs = Float64[]
    t = 0.0
    for i in 1:nseg
        τ = args.args[i * 5 - 4]
        Ω = args.args[i * 5 - 3]
        Ω′ = args.args[i * 5 - 2]

        push!(ts, t)
        push!(Ωs, Ω)

        t += τ
        push!(ts, t)
        push!(Ωs, Ω + Ω′ * τ)
    end
    return ts, Ωs
end

function get_φs(args::RawParams; δ=0.0)
    nargs = length(args.args)
    @assert nargs % 5 == 0
    nseg = nargs ÷ 5

    ts = Float64[]
    φs = Float64[]
    t = 0.0
    for i in 1:nseg
        τ = args.args[i * 5 - 4]
        φ = args.args[i * 5 - 1]
        ω = args.args[i * 5]

        push!(ts, t)
        push!(φs, φ + δ * t)

        t += τ
        push!(ts, t)
        push!(φs, φ + ω * τ + δ * t)
    end
    return ts, φs
end

function get_ωs(args::RawParams)
    nargs = length(args.args)
    @assert nargs % 5 == 0
    nseg = nargs ÷ 5

    ts = Float64[]
    ωs = Float64[]
    t = 0.0
    for i in 1:nseg
        τ = args.args[i * 5 - 4]
        ω = args.args[i * 5]

        push!(ts, t)
        push!(ωs, ω)

        t += τ
        push!(ts, t)
        push!(ωs, ω)
    end
    return ts, ωs
end

struct ModeInfo{Kern}
    kern::Kern
    ωm::Float64
    weight::Float64
end

struct Objective{pmask,ObjArg,NSeg,Param,Obj,Modes<:Tuple,NArgs,NObjArgs} <: Function
    modes::Modes
    param::Param
    obj::Obj
    args::MVector{NArgs,Float64}
    grads::MVector{NArgs,Float64}
    objargs::MVector{NObjArgs,Float64}
    objgrads::MVector{NObjArgs,Float64}
end

nparams(m::Objective) = nparams(m.param)

function Objective(pmask, ObjArg, obj::Obj, modes::Modes,
                   buf::SL.ComputeBuffer{NSeg};
                   freq=FreqSpec(), amp=AmpSpec()) where {NSeg,Obj}
    modes_array = []
    for (modei, (ωm, weight)) in enumerate(modes.modes)
        kern = SL.Kernel(buf, Val(pmask))
        push!(modes_array, ModeInfo(kern, ωm, weight))
    end
    modes = (modes_array...,)
    param = ModSpec{NSeg}(freq=freq, amp=amp)
    NArgs = NSeg * 5
    NObjArgs = length(ObjArg)
    return Objective{pmask,ObjArg,NSeg,typeof(param),Obj,typeof(modes),NArgs,
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

function _generate_nlobj(ObjArg, NSeg, Modes, obj_ex, grads_out_var, objargs_ex, objgrads_ex)
    nmodes = length(Modes.parameters)

    # 1. Forward propagation of values
    # input parameter -> pulse parameter
    func_ex = quote
        args = m.args
        grads = m.grads
        objargs = $objargs_ex
        objgrads = $objgrads_ex
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
                  SL.eval_with_mode!($(mode_var.kern), args, $(mode_var.ωm))
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
            else
                :(error("Invalid objective argument ", $(QuoteNode(key))))
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
        else
            :(error("Invalid objective argument ", $(QuoteNode(key))))
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
                  return $(obj_ex)(objargs)
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

const _VecOrTup = Union{AbstractVector,Tuple}
const _VecOrTup0 = Union{AbstractVector,Tuple{}}

@generated function (m::Objective{pmask,ObjArg,NSeg,Param,Obj,Modes})(
    x::_VecOrTup, grads_out::_VecOrTup0) where {pmask,ObjArg,NSeg,Param,Obj,Modes}
    return _generate_nlobj(ObjArg, NSeg, Modes, :(m.obj), :grads_out,
                           :(m.objargs), :(m.objgrads))
end

function RawParams(m::Objective{pmask,ObjArg,NSeg}, x;
                   buff=Vector{Float64}(undef, NSeg * 5)) where {pmask,ObjArg,NSeg}
    transform_argument(m.param, buff, x)
    return RawParams(buff)
end

@inline _dummy_obj(x) = x[1]

function _process_objarg(ObjArg::Tuple)
    if ObjArg === () || isa(ObjArg[1], Tuple)
        return ObjArg
    end
    return (ObjArg,)
end

@generated function (m::Objective{pmask,ObjArg,NSeg,Param,Obj,Modes})(
    ::Val{ObjArg2}, x::_VecOrTup) where {pmask,ObjArg,NSeg,Param,Obj,Modes,ObjArg2}
    ObjArg2 = _process_objarg(ObjArg2)
    @assert length(ObjArg2) == 1
    return _generate_nlobj(ObjArg2, NSeg, Modes, :_dummy_obj, nothing,
                           :(MVector{$(length(ObjArg2)),Float64}(undef)), nothing)
end

@generated function (m::Objective{pmask,ObjArg,NSeg,Param,Obj,Modes})(
    cb, ::Val{ObjArg2}, x::_VecOrTup) where {pmask,ObjArg,NSeg,Param,Obj,Modes,ObjArg2}
    ObjArg2 = _process_objarg(ObjArg2)
    return _generate_nlobj(ObjArg2, NSeg, Modes, :cb, nothing,
                           :(MVector{$(length(ObjArg2)),Float64}(undef)), nothing)
end

struct SolutionProperties
    total_time::Float64
    modes::Vector{Float64}
    dis::Vector{ComplexF64}
    disδ::Vector{ComplexF64}
    cumdis::Vector{ComplexF64}
    area::Vector{Float64}
    areaδ::Vector{Float64}
end

Base.Dict(props::SolutionProperties) = Dict(
    "total_time"=>props.total_time,
    "modes"=>props.modes,
    "rdis"=>real.(props.dis),
    "idis"=>imag.(props.dis),
    "rdisδ"=>real.(props.disδ),
    "idisδ"=>imag.(props.disδ),
    "rcumdis"=>real.(props.cumdis),
    "icumdis"=>imag.(props.cumdis),
    "area"=>props.area,
    "areaδ"=>props.areaδ,
)

SolutionProperties(d::Dict{<:AbstractString}) = SolutionProperties(
    d["total_time"],
    copy(d["modes"]),
    complex.(d["rdis"], d["idis"]),
    complex.(d["rdisδ"], d["idisδ"]),
    complex.(d["rcumdis"], d["icumdis"]),
    copy(d["area"]),
    copy(d["areaδ"]),
)

struct Summarizer{NSeg,Kern}
    kern::Kern
    function Summarizer{NSeg}() where NSeg
        mask = SS.ValueMask(true, true, true, true, true, true)
        buf = SL.ComputeBuffer{NSeg,Float64}(Val(mask), Val(zero(SS.ValueMask)))
        kern = SL.Kernel(buf, Val(zero(SL.ParamGradMask)))
        return new{NSeg,typeof(kern)}(kern)
    end
end

function Base.get(s::Summarizer{NSeg}, params::RawParams, modes::Modes) where NSeg
    kern = s.kern
    args = params.args
    @assert length(args) == NSeg * 5
    total_time = sum(@view args[1:5:end])
    info = SolutionProperties(total_time, Float64[],
                              ComplexF64[], ComplexF64[], ComplexF64[],
                              Float64[], Float64[])
    for (ωm, _) in modes.modes
        SL.eval_with_mode!(kern, args, ωm)
        val = kern.result.val
        push!(info.modes, ωm)
        push!(info.dis, val.dis)
        push!(info.disδ, val.disδ)
        push!(info.cumdis, val.cumdis)
        push!(info.area, val.area)
        push!(info.areaδ, val.areaδ)
    end
    return info
end

end
