#!/usr/bin/julia

module SymLinear

using StaticArrays

# Integral for pulses that are piecewise linear in both amplitude and phase.

module SegInt

import ...Utils
import ...SegSeq

using StaticArrays

# Generate a structure with the trig ratios we need precomputed
# It's easier for the compiler to do DCE on the values we don't use
# then to do CSE on branchy code that are duplicated.
function _trig_field_name(name)
    m = match(r"^sin_c([_0-9]*)$", name)
    if m !== nothing
        return Symbol("S_C$(m[1])")
    end
    m = match(r"^cos_c([_0-9]*)$", name)
    if m !== nothing
        return Symbol("C_C$(m[1])")
    end
    m = match(r"^sin_f([_0-9]*)$", name)
    if m !== nothing
        return Symbol("S$(m[1])")
    end
    m = match(r"^cos_f([_0-9]*)$", name)
    if m !== nothing
        return Symbol("C$(m[1])")
    end
end

macro gen_trig_ratios(d, s, c, names...)
    expr = :(())
    for (name::Symbol) in names
        field_name = _trig_field_name(String(name))
        if field_name === nothing
            error("Unknown function name $name")
        end
        push!(expr.args, :($field_name = Utils.$name($(esc(d)), $(esc(s)), $(esc(c)))))
    end
    return expr
end

# Integral for each segments
# The kernel version are shared by both the test version
# and the version used in actual computation.
@inline function displacement_kernel(o, o′, d, s, c, V)
    return complex(muladd(o + o′, V.S_C1, -o′ * V.C1),
                   muladd(o * d, V.C1, o′ * V.S_C2))
end

@inline function displacement_δ_kernel(o, o′, d, s, c, V)
    return complex(muladd(o + o′, -V.S_C2, o′ * V.C2),
                   muladd(o, muladd(-d, V.C2, V.C1), o′ * V.S_C3))
end

@inline function displacement_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (complex(muladd(Ω′, τ, Ω) * c, muladd(Ω′, τ, Ω) * s),
            τ * complex(V.S_C1, d * V.C1),
            τ^2 * complex(V.S_C1 - V.C1, V.S_C2))
end

@inline function displacement_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return ((o + o′) * complex(-s, c),
            τ^2 * complex(-V.S_C2, muladd(d, -V.C2, V.C1)),
            τ^2 * τ * complex(V.C2 - V.S_C2, V.S_C3),
            τ^2 * complex(muladd(o + o′, -V.S_C3, o′ * V.C3_2),
                           -muladd(o′, V.S3_3, o * muladd(d, V.C3_2, 2 * V.C2))))
end

@inline function cumulative_displacement_kernel(o, o′, d, s, c, V)
    return complex(muladd(o, V.C1, o′ * V.S2), muladd(o, V.S1 * d, o′ * V.C2))
end

@inline function cumulative_displacement_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (complex(muladd(o + o′, V.S_C1, -(o′ * V.C1)),
                    muladd(o * d, V.C1, o′ * V.S_C2)),
            τ^2 * complex(V.C1, V.S1 * d), τ * τ^2 * complex(V.S2, V.C2),
            τ^2 * complex(-muladd(o, V.C2, o′ * V.S3_2),
                           muladd(o, V.S2, o′ * V.C3_2)))
end

# Twice the enclosed area
@inline function enclosed_area_complex_kernel(o, o′, d, s, c, V)
    a1 = o * (o + o′)
    a2 = o′^2
    return complex(muladd(a1, V.C1, a2 * V.C3), muladd(a1, V.S1 * d, a2 * V.S3))
end

# Twice the enclosed area
@inline function enclosed_area_kernel(o, o′, d, s, c, V)
    a1 = o * (o + o′)
    a2 = o′^2
    return muladd(a1, V.S1 * d, a2 * V.S3)
end

@inline function enclosed_area_δ_kernel(o, o′, d, s, c, V)
    a1 = o * (o + o′)
    a2 = o′^2
    return muladd(a1, V.S2, a2 * V.S4)
end

@inline function enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (muladd(Ω′, τ, Ω) * d * muladd(o, V.C1, o′ * V.S1),
            τ * muladd(2, o, o′) * (V.S1 * d),
            τ^2 * muladd(2 * o′, V.S3, o * (V.S1 * d)))
end

@inline function enclosed_area_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (muladd(muladd(muladd(-2, V.S1, V.S_C1), o′, o * (V.S_C1 - V.C1)),
                   o, o′^2 * V.S2),
            τ^2 * muladd(2, o, o′) * V.S2,
            τ^2 * τ * muladd(2 * o′, V.S4, o * V.S2),
            τ^2 * muladd(o * (o + o′), -V.S3_2, -o′^2 * V.S5))
end

# These are for testing only.
# The `compute_values` below is the one that's used in actual computation.
function displacement(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_c1, sin_c2, cos_f1)
    return phase0 * displacement_kernel(o, o′, d, s, c, V)
end

function displacement_δ(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, cos_f2, sin_c2, sin_c3)
    return phase0 * τ * displacement_δ_kernel(o, o′, d, s, c, V)
end

function displacement_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_c1, sin_c2, sin_c3, cos_f1, cos_f2)

    τΩs = displacement_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (phase0 * τΩs[1], phase0 * τΩs[2], phase0 * τΩs[3],
            Utils.mulim(phase0 * displacement_kernel(o, o′, d, s, c, V)),
            phase0 * τ * displacement_δ_kernel(o, o′, d, s, c, V))
end

function displacement_δ_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, cos_f2, cos_f3_2, sin_f3_3, sin_c2, sin_c3)

    τΩsδ = displacement_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (phase0 * τΩsδ[1], phase0 * τΩsδ[2], phase0 * τΩsδ[3],
            Utils.mulim(phase0 * τ * displacement_δ_kernel(o, o′, d, s, c, V)),
            phase0 * τΩsδ[4])
end

function cumulative_displacement(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, cos_f2, sin_f2)

    return phase0 * τ * cumulative_displacement_kernel(o, o′, d, s, c, V)
end

function cumulative_displacement_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, cos_f2, sin_f2,
                         sin_f3_2, cos_f3_2, sin_c1, sin_c2)

    τΩsδ = cumulative_displacement_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (phase0 * τΩsδ[1], phase0 * τΩsδ[2], phase0 * τΩsδ[3],
            Utils.mulim(phase0 * τ * cumulative_displacement_kernel(o, o′, d, s, c, V)),
            phase0 * τΩsδ[4])
end

# Twice the enclosed area
function enclosed_area_complex(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, cos_f3, sin_f3)

    return enclosed_area_complex_kernel(o, o′, d, s, c, V)
end

# Twice the enclosed area
function enclosed_area(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_f1, sin_f3)

    return enclosed_area_kernel(o, o′, d, s, c, V)
end

function enclosed_area_δ(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_f2, sin_f4)

    return τ * enclosed_area_δ_kernel(o, o′, d, s, c, V)
end

function enclosed_area_gradients(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, sin_f2, sin_f3, sin_f4)

    τΩs = enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (τΩs[1], τΩs[2], τΩs[3], zero(φ),
            τ * enclosed_area_δ_kernel(o, o′, d, s, c, V))
end

function enclosed_area_δ_gradients(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = Utils.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, sin_f2, sin_f3_2,
                         sin_f4, sin_f5, sin_c1)

    τΩsδ = enclosed_area_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (τΩsδ[1], τΩsδ[2], τΩsδ[3], zero(φ), τΩsδ[4])
end

# The values we may care about in each segments
# * Displacement (dis)
# * Gradient of displacement w.r.t. detuning (disδ)
# * Cumulative displacement (cumdis)
# * Enclosed area (area)
# * Gradient of enclosed area w.r.t. detuning (areaδ)
# As well as the gradient of everything above w.r.t. each of the input parameters

# Compute all the values we want for this segment in one go.
# This should allow the compiler to reuse many of the intermediate results
# when computing different values.
@inline function (compute_values(τ::_T, Ω, Ω′, φ, δ, ::Val{maskv}, ::Val{maskg})
                  where {_T,maskv,maskg})

    T = float(_T)
    CT = Complex{T}
    SDV = SegSeq.SegData(T, maskv)
    SDG = SegSeq.SegData(T, maskg)

    need_grad = maskg !== zero(SegSeq.ValueMask)

    @inline begin
        d = δ * τ
        o = Ω * τ
        o′ = Ω′ * τ^2
        s, c = Utils.fast_sincos(d)
        sφ, cφ = Utils.fast_sincos(φ)
        phase0 = complex(cφ, sφ)
        phase0_τ = phase0 * τ
        V = @gen_trig_ratios(d, s, c, sin_c1, sin_c2, sin_c3,
                             cos_f1, sin_f1, cos_f2, sin_f2,
                             cos_f3_2, sin_f3, sin_f3_2, sin_f3_3, sin_f4, sin_f5)

        dis = Utils.mul(phase0, displacement_kernel(o, o′, d, s, c, V))
        area = enclosed_area_kernel(o, o′, d, s, c, V)
        cumdis = Utils.mul(phase0_τ, cumulative_displacement_kernel(o, o′, d, s, c, V))
        disδ = Utils.mul(phase0_τ, displacement_δ_kernel(o, o′, d, s, c, V))
        areaδ = τ * enclosed_area_δ_kernel(o, o′, d, s, c, V)
        res = SDV(maskv.τ ? τ : nothing, maskv.dis ? dis : nothing,
                  maskv.area ? area : nothing, maskv.cumdis ? cumdis : nothing,
                  maskv.disδ ? disδ : nothing, maskv.areaδ ? areaδ : nothing)
        if !need_grad
            return res, nothing
        end
        if maskg.τ
            τ_grad = SA[one(T), zero(T), zero(T), zero(T), zero(T)]
        else
            τ_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        if maskg.dis
            dis_τΩs = displacement_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
            dis_grad = SA[Utils.mul(phase0, dis_τΩs[1]), Utils.mul(phase0, dis_τΩs[2]),
                          Utils.mul(phase0, dis_τΩs[3]), Utils.mulim(dis), disδ]
        else
            dis_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        if maskg.area
            area_τΩs = enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
            area_grad = SA[area_τΩs[1], area_τΩs[2], area_τΩs[3], zero(T), areaδ]
        else
            area_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        if maskg.cumdis
            cumdis_τΩsδ = cumulative_displacement_τΩsδ_kernel(o, o′, d, s, c,
                                                                    Ω, Ω′, τ, V)
            cumdis_grad = SA[Utils.mul(phase0, cumdis_τΩsδ[1]),
                             Utils.mul(phase0, cumdis_τΩsδ[2]),
                             Utils.mul(phase0, cumdis_τΩsδ[3]),
                             Utils.mulim(cumdis),
                             Utils.mul(phase0, cumdis_τΩsδ[4])]
        else
            cumdis_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        if maskg.disδ
            disδ_τΩsδ = displacement_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
            disδ_grad = SA[Utils.mul(phase0, disδ_τΩsδ[1]),
                            Utils.mul(phase0, disδ_τΩsδ[2]),
                            Utils.mul(phase0, disδ_τΩsδ[3]),
                            Utils.mulim(disδ),
                            Utils.mul(phase0, disδ_τΩsδ[4])]
        else
            disδ_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        if maskg.areaδ
            areaδ_τΩsδ = enclosed_area_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
            areaδ_grad = SA[areaδ_τΩsδ[1], areaδ_τΩsδ[2], areaδ_τΩsδ[3],
                             zero(T), areaδ_τΩsδ[4]]
        else
            areaδ_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        grads = SA[SDG(τ_grad[1], dis_grad[1], area_grad[1], cumdis_grad[1],
                       disδ_grad[1], areaδ_grad[1]),
                   SDG(τ_grad[2], dis_grad[2], area_grad[2], cumdis_grad[2],
                       disδ_grad[2], areaδ_grad[2]),
                   SDG(τ_grad[3], dis_grad[3], area_grad[3], cumdis_grad[3],
                       disδ_grad[3], areaδ_grad[3]),
                   SDG(τ_grad[4], dis_grad[4], area_grad[4], cumdis_grad[4],
                       disδ_grad[4], areaδ_grad[4]),
                   SDG(τ_grad[5], dis_grad[5], area_grad[5], cumdis_grad[5],
                       disδ_grad[5], areaδ_grad[5])]
        return res, grads
    end
end

end

import ..Utils
import ..SegSeq

struct ParamGradMask
    τ::Bool
    Ω::Bool
    Ω′::Bool
    φ::Bool
    ω::Bool
end

@inline function grad_index(param_mask::ParamGradMask, name::Symbol)
    idx = 1
    if name === :τ
        return param_mask.τ ? idx : 0
    elseif param_mask.τ
        idx += 1
    end
    if name === :dΩ
        return param_mask.Ω ? idx : 0
    elseif param_mask.Ω
        idx += 1
    end
    if name === :Ω′
        return param_mask.Ω′ ? idx : 0
    elseif param_mask.Ω′
        idx += 1
    end
    if name === :dφ
        return param_mask.φ ? idx : 0
    elseif param_mask.φ
        idx += 1
    end
    if name === :ω
        return param_mask.ω ? idx : 0
    elseif param_mask.ω
        idx += 1
    end
    return 0
end

# Segment parameter
# We use the phase and amplitude jump instead of their absolute values
# as the parameters since these are closer to what we actually want to control
# in the experiment.
struct Pulse{T}
    τ::T
    dΩ::T
    Ω′::T
    dφ::T
    ω::T
end

struct Mode{T}
    ω::T
    dis_scale::T
    area_scale::T
end

mutable struct System{T,SDV<:SegSeq.SegData{T},SDG<:SegSeq.SegData{T},MR,pmask}
    const pulses::Vector{Pulse{T}}
    const modes::Vector{Mode{T}}

    cur_mod::Int
    const seg_buf::Vector{SDV}
    const seg_grad_buf::Utils.JaggedMatrix{SDG}
    const buffer::SegSeq.SeqComputeBuffer{T}
    const single_result::SegSeq.SingleModeResult{T,SDV,SDG}
    const result::MR

    function System{T}(modes::AbstractVector, ::Val{maskv}, ::Val{maskg},
                       ::Val{pmask}) where {T,maskv,maskg,pmask}

        SDV = SegSeq.SegData(T, maskv)
        SDG = SegSeq.SegData(T, maskg)

        pulses = Pulse{T}[]
        seg_buf = SDV[]
        seg_grad_buf = Utils.JaggedMatrix{SDG}()
        buffer = SegSeq.SeqComputeBuffer{T}()
        single_result = SegSeq.SingleModeResult{T}(Val(maskv), Val(maskg))
        result = SegSeq.MultiModeResult{T}(Val(maskv), Val(maskg))
        return new{T,SDV,SDG,typeof(result),pmask}(pulses, modes, 0, seg_buf,
                                                   seg_grad_buf, buffer,
                                                   single_result, result)
    end
    function System{T}(::Val{maskv}, ::Val{maskg},
                       ::Val{pmask}) where {T,maskv,maskg,pmask}
        return System{T}(Mode{T}[], Val(maskv), Val(maskg), Val(pmask))
    end
end

@inline function pindex_input(pmask::ParamGradMask)
    ngrad_in = 0
    idxs = (τ=pmask.τ ? (ngrad_in = ngrad_in + 1) : nothing,
            Ω=(pmask.Ω || pmask.Ω′) ? (ngrad_in = ngrad_in + 1) : nothing,
            Ω′=pmask.Ω′ ? (ngrad_in = ngrad_in + 1) : nothing,
            φ=(pmask.φ || pmask.ω) ? (ngrad_in = ngrad_in + 1) : nothing,
            δ=pmask.ω ? (ngrad_in = ngrad_in + 1) : nothing)
    return idxs, ngrad_in
end
@inline function pindex_output(pmask::ParamGradMask)
    ngrad_out = 0
    idxs = (τ=pmask.τ ? (ngrad_out = ngrad_out + 1) : nothing,
            Ω=pmask.Ω ? (ngrad_out = ngrad_out + 1) : nothing,
            Ω′=pmask.Ω′ ? (ngrad_out = ngrad_out + 1) : nothing,
            φ=pmask.φ ? (ngrad_out = ngrad_out + 1) : nothing,
            δ=pmask.ω ? (ngrad_out = ngrad_out + 1) : nothing)
    return idxs, ngrad_out
end

@inline function _fill_seg_buf!(sys::System{T,SDV,SDG,MR,pmask},
                                mode_idx) where {T,SDV,SDG,MR,pmask}

    maskv = SegSeq.value_mask(SDV)
    maskg = SegSeq.value_mask(SDG)

    nseg = length(sys.pulses)

    @inbounds mode = sys.modes[mode_idx]
    φ = zero(T)
    Ω = zero(T)
    pulses = sys.pulses
    seg_buf = sys.seg_buf
    seg_grad_buf = sys.seg_grad_buf
    @inline @inbounds for i in 1:nseg
        pulse = pulses[i]
        Ω += pulse.dΩ
        φ += pulse.dφ
        δ = pulse.ω - mode.ω
        if pulse.Ω′ == 0
            seg, grad = SegInt.compute_values(pulse.τ, Ω, Utils.Zero(),
                                              φ, δ, Val(maskv), Val(maskg))
        else
            seg, grad = SegInt.compute_values(pulse.τ, Ω, pulse.Ω′,
                                              φ, δ, Val(maskv), Val(maskg))
        end
        seg_buf[i] = seg
        if maskg != zero(SegSeq.ValueMask)
            grad_buf = seg_grad_buf[i]
            gi, = pindex_input(pmask)
            pmask.τ && (grad_buf[gi.τ] = grad[1])
            (pmask.Ω || pmask.Ω′) && (grad_buf[gi.Ω] = grad[2])
            pmask.Ω′ && (grad_buf[gi.Ω′] = grad[3])
            (pmask.φ || pmask.ω) && (grad_buf[gi.φ] = grad[4])
            pmask.ω && (grad_buf[gi.δ] = grad[5])
        end
        φ = muladd(pulse.τ, δ, φ)
        Ω = muladd(pulse.τ, pulse.Ω′, Ω)
    end
end

@inline function _init_grads_vector(grads::Vector{Utils.JaggedMatrix{T}},
                                    nmodes, nseg, ngrad) where T
    resize!(grads, nmodes)
    @inbounds for i in 1:nmodes
        if !isassigned(grads, i)
            grads[i] = Utils.JaggedMatrix{T}()
        end
        Utils.resize_uniform!(grads[i], nseg, ngrad)
    end
    return
end

@inline _compute_mode!(sys::System{T,SDV,SDG,MR,pmask}, mode_idx, single_result) where {T,SDV,SDG,MR,pmask} = @inline @inbounds begin
    maskv = SegSeq.value_mask(SDV)
    maskg = SegSeq.value_mask(SDG)
    need_grad = maskg !== zero(SegSeq.ValueMask)

    gi, ngrad_in = pindex_input(pmask)
    go, ngrad_out = pindex_output(pmask)

    pulses = sys.pulses
    nseg = length(pulses)
    result = sys.result

    _fill_seg_buf!(sys, mode_idx)
    mode = sys.modes[mode_idx]
    SegSeq.compute_single_mode!(single_result, sys.seg_buf, sys.buffer,
                                need_grad ? sys.seg_grad_buf : nothing,
                                Val(pmask.τ), Val(true))

    dis_scale = mode.dis_scale
    area_scale = mode.area_scale

    if mode_idx == 1
        result.τ = single_result.val.τ
    end
    maskv.dis && (result.dis[mode_idx] = dis_scale * single_result.val.dis)
    maskv.area && (result.area = muladd(area_scale, single_result.val.area,
                                        result.area))
    maskv.cumdis && (result.cumdis[mode_idx] = dis_scale * single_result.val.cumdis)
    maskv.disδ && (result.disδ[mode_idx] = dis_scale * single_result.val.disδ)
    maskv.areaδ && (result.areaδ[mode_idx] = area_scale * single_result.val.areaδ)

    if !need_grad
        return
    end

    if mode_idx == 1
        if maskg.τ
            for i in 1:length(result.τ_grad.values)
                result.τ_grad.values[i] = single_result.grad.values[i].τ
            end
        end
    end
    if maskg.dis
        dis_grad = result.dis_grad[mode_idx]
    end
    if maskg.cumdis
        cumdis_grad = result.cumdis_grad[mode_idx]
    end
    if maskg.disδ
        disδ_grad = result.disδ_grad[mode_idx]
    end
    if maskg.areaδ
        areaδ_grad = result.areaδ_grad[mode_idx]
    end

    dis_dΩ = complex(zero(T))
    area_dΩ = zero(T)
    cumdis_dΩ = complex(zero(T))
    areaδ_dΩ = zero(T)
    disδ_dΩ = complex(zero(T))

    dis_dφ = complex(zero(T))
    area_dφ = zero(T)
    cumdis_dφ = complex(zero(T))
    areaδ_dφ = zero(T)
    disδ_dφ = complex(zero(T))

    for seg_idx in nseg:-1:1
        pulse = pulses[seg_idx]
        δ = pulse.ω - mode.ω

        sgrad = single_result.grad[seg_idx]

        # displacement
        if maskg.dis
            res_dis_sgrad = dis_grad[seg_idx]
            # τ
            if pmask.τ
                res_dis_sgrad[go.τ] = dis_scale * muladd(dis_dφ, δ,
                                                          muladd(dis_dΩ, pulse.Ω′,
                                                                 sgrad[gi.τ].dis))
            end
            # Ω
            if pmask.Ω || pmask.Ω′
                old_dis_dΩ = dis_dΩ
                dis_dΩ = sgrad[gi.Ω].dis + dis_dΩ
            end
            if pmask.Ω
                res_dis_sgrad[go.Ω] = dis_scale * dis_dΩ
            end
            # Ω′
            if pmask.Ω′
                res_dis_sgrad[go.Ω′] = dis_scale * muladd(old_dis_dΩ, pulse.τ,
                                                            sgrad[gi.Ω′].dis)
            end
            # φ
            if pmask.φ || pmask.ω
                old_dis_dφ = dis_dφ
                dis_dφ = sgrad[gi.φ].dis + dis_dφ
            end
            if pmask.φ
                res_dis_sgrad[go.φ] = dis_scale * dis_dφ
            end
            # ω
            if pmask.ω
                res_dis_sgrad[go.δ] = dis_scale * muladd(old_dis_dφ, pulse.τ,
                                                          sgrad[gi.δ].dis)
            end
        end

        # area
        if maskg.area
            res_area_sgrad = result.area_grad[seg_idx]
            # τ
            if pmask.τ
                res_area_sgrad[go.τ] = muladd(muladd(area_dφ, δ,
                                                      muladd(area_dΩ, pulse.Ω′,
                                                             sgrad[gi.τ].area)),
                                               area_scale, res_area_sgrad[go.τ])
            end
            # Ω
            if pmask.Ω || pmask.Ω′
                old_area_dΩ = area_dΩ
                area_dΩ = sgrad[gi.Ω].area + area_dΩ
            end
            if pmask.Ω
                res_area_sgrad[go.Ω] = muladd(area_scale, area_dΩ,
                                               res_area_sgrad[go.Ω])
            end
            # Ω′
            if pmask.Ω′
                res_area_sgrad[go.Ω′] = muladd(muladd(old_area_dΩ, pulse.τ,
                                                        sgrad[gi.Ω′].area),
                                                 area_scale,
                                                 res_area_sgrad[go.Ω′])
            end
            # φ
            if pmask.φ || pmask.ω
                old_area_dφ = area_dφ
                area_dφ = sgrad[gi.φ].area + area_dφ
            end
            if pmask.φ
                res_area_sgrad[go.φ] = muladd(area_scale, area_dφ,
                                               res_area_sgrad[go.φ])
            end
            # ω
            if pmask.ω
                res_area_sgrad[go.δ] = muladd(muladd(old_area_dφ, pulse.τ,
                                                      sgrad[gi.δ].area),
                                               area_scale, res_area_sgrad[go.δ])
            end
        end

        if maskg.cumdis
            res_cumdis_sgrad = cumdis_grad[seg_idx]
            # τ
            if pmask.τ
                res_cumdis_sgrad[go.τ] = dis_scale * muladd(
                    cumdis_dφ, δ,
                    muladd(cumdis_dΩ, pulse.Ω′, sgrad[gi.τ].cumdis))
            end
            # Ω
            if pmask.Ω || pmask.Ω′
                old_cumdis_dΩ = cumdis_dΩ
                cumdis_dΩ = sgrad[gi.Ω].cumdis + cumdis_dΩ
            end
            if pmask.Ω
                res_cumdis_sgrad[go.Ω] = dis_scale * cumdis_dΩ
            end
            # Ω′
            if pmask.Ω′
                res_cumdis_sgrad[go.Ω′] =
                    dis_scale * muladd(old_cumdis_dΩ, pulse.τ,
                                       sgrad[gi.Ω′].cumdis)
            end
            # φ
            if pmask.φ || pmask.ω
                old_cumdis_dφ = cumdis_dφ
                cumdis_dφ = sgrad[gi.φ].cumdis + cumdis_dφ
            end
            if pmask.φ
                res_cumdis_sgrad[go.φ] = dis_scale * cumdis_dφ
            end
            # ω
            if pmask.ω
                res_cumdis_sgrad[go.δ] = dis_scale * muladd(old_cumdis_dφ, pulse.τ,
                                                             sgrad[gi.δ].cumdis)
            end
        end

        # displacement mode gradient
        if maskg.disδ
            res_disδ_sgrad = disδ_grad[seg_idx]
            # τ
            if pmask.τ
                res_disδ_sgrad[go.τ] = dis_scale * muladd(
                    disδ_dφ, δ, muladd(disδ_dΩ, pulse.Ω′, sgrad[gi.τ].disδ))
            end
            # Ω
            if pmask.Ω || pmask.Ω′
                old_disδ_dΩ = disδ_dΩ
                disδ_dΩ = sgrad[gi.Ω].disδ + disδ_dΩ
            end
            if pmask.Ω
                res_disδ_sgrad[go.Ω] = dis_scale * disδ_dΩ
            end
            # Ω′
            if pmask.Ω′
                res_disδ_sgrad[go.Ω′] = dis_scale * muladd(old_disδ_dΩ, pulse.τ,
                                                              sgrad[gi.Ω′].disδ)
            end
            # φ
            if pmask.φ || pmask.ω
                old_disδ_dφ = disδ_dφ
                disδ_dφ = sgrad[gi.φ].disδ + disδ_dφ
            end
            if pmask.φ
                res_disδ_sgrad[go.φ] = dis_scale * disδ_dφ
            end
            # ω
            if pmask.ω
                res_disδ_sgrad[go.δ] = dis_scale * muladd(old_disδ_dφ, pulse.τ,
                                                            sgrad[gi.δ].disδ)
            end
        end

        # area mode gradient
        if maskg.areaδ
            res_areaδ_sgrad = areaδ_grad[seg_idx]
            # τ
            if pmask.τ
                res_areaδ_sgrad[go.τ] = area_scale * muladd(
                    areaδ_dφ, δ,
                    muladd(areaδ_dΩ, pulse.Ω′, sgrad[gi.τ].areaδ))
            end
            # Ω
            if pmask.Ω || pmask.Ω′
                old_areaδ_dΩ = areaδ_dΩ
                areaδ_dΩ = sgrad[gi.Ω].areaδ + areaδ_dΩ
            end
            if pmask.Ω
                res_areaδ_sgrad[go.Ω] = area_scale * areaδ_dΩ
            end
            # Ω′
            if pmask.Ω′
                res_areaδ_sgrad[go.Ω′] = area_scale * muladd(old_areaδ_dΩ, pulse.τ,
                                                                sgrad[gi.Ω′].areaδ)
            end
            # φ
            if pmask.φ || pmask.ω
                old_areaδ_dφ = areaδ_dφ
                areaδ_dφ = sgrad[gi.φ].areaδ + areaδ_dφ
            end
            if pmask.φ
                res_areaδ_sgrad[go.φ] = area_scale * areaδ_dφ
            end
            # ω
            if pmask.ω
                res_areaδ_sgrad[go.δ] = area_scale * muladd(old_areaδ_dφ, pulse.τ,
                                                              sgrad[gi.δ].areaδ)
            end
        end
    end
end

function compute!(sys::System{T,SDV,SDG,MR,pmask}) where {T,SDV,SDG,MR,pmask}

    maskv = SegSeq.value_mask(SDV)
    maskg = SegSeq.value_mask(SDG)
    need_grad = maskg !== zero(SegSeq.ValueMask)

    gi, ngrad_in = pindex_input(pmask)
    go, ngrad_out = pindex_output(pmask)

    pulses = sys.pulses
    nmodes = length(sys.modes)
    nseg = length(pulses)

    result = sys.result
    single_result = sys.single_result

    maskv.dis && resize!(result.dis, nmodes)
    maskv.area && (result.area = zero(T))
    maskv.cumdis && resize!(result.cumdis, nmodes)
    maskv.disδ && resize!(result.disδ, nmodes)
    maskv.areaδ && resize!(result.areaδ, nmodes)

    maskg.τ && Utils.resize_uniform!(result.τ_grad, nseg, ngrad_out)
    maskg.dis && _init_grads_vector(result.dis_grad, nmodes, nseg, ngrad_out)
    if maskg.area
        Utils.resize_uniform!(result.area_grad, nseg, ngrad_out)
        @inbounds result.area_grad.values .= 0
    end
    maskg.cumdis && _init_grads_vector(result.cumdis_grad, nmodes, nseg, ngrad_out)
    maskg.disδ && _init_grads_vector(result.disδ_grad, nmodes, nseg, ngrad_out)
    maskg.areaδ && _init_grads_vector(result.areaδ_grad, nmodes, nseg, ngrad_out)

    resize!(sys.seg_buf, nseg)
    if need_grad
        Utils.resize_uniform!(sys.seg_grad_buf, nseg, ngrad_in)
        Utils.resize_uniform!(single_result.grad, nseg, ngrad_in)
    else
        empty!(single_result.grad)
    end

    @inline @inbounds for mode_idx in 1:nmodes
        _compute_mode!(sys, mode_idx, single_result)
    end
    return
end

struct ComputeBuffer{NSeg,T,SDV<:SegSeq.SegData{T},SDG<:SegSeq.SegData{T}}
    seg_buf::Vector{SDV}
    seg_grad_buf::Utils.JaggedMatrix{SDG}
    buffer::SegSeq.SeqComputeBuffer{T}

    function ComputeBuffer{NSeg,T}(::Val{maskv}, ::Val{maskg}) where {NSeg,T,maskv,maskg}
        SDV = SegSeq.SegData(T, maskv)
        SDG = SegSeq.SegData(T, maskg)

        seg_buf = Vector{SDV}(undef, NSeg)
        seg_grad_buf = Utils.JaggedMatrix{SDG}()
        buffer = SegSeq.SeqComputeBuffer{T}()
        Utils.resize_uniform!(seg_grad_buf, NSeg, 5)
        return new{NSeg,T,SDV,SDG}(seg_buf, seg_grad_buf, buffer)
    end
end

mutable struct Kernel{NSeg,T,SDV<:SegSeq.SegData{T},SDG<:SegSeq.SegData{T},pmask,NArgs}
    const buffer::ComputeBuffer{NSeg,T,SDV,SDG}
    const result::SegSeq.SingleModeResult{T,SDV,SDG}
    evaled::Bool
    args::NTuple{NArgs,T}

    function Kernel(buffer::ComputeBuffer{NSeg,T,SDV,SDG},
                    ::Val{pmask}) where {NSeg,T,SDV,SDG,pmask}
        maskv = SegSeq.value_mask(SDV)
        maskg = SegSeq.value_mask(SDG)
        result = SegSeq.SingleModeResult{T}(Val(maskv), Val(maskg))
        Utils.resize_uniform!(result.grad, NSeg, 5)
        return new{NSeg,T,SDV,SDG,pmask,NSeg*5}(buffer, result, false)
    end
end

# Arguments
# (τ, Ω, Ω′, φ, δ) * nseg
function _update!(kern::Kernel{NSeg,T,SDV,SDG,pmask,NArgs}) where {NSeg,T,SDV,SDG,pmask,NArgs}
    buffer = kern.buffer
    maskv = SegSeq.value_mask(SDV)
    maskg = SegSeq.value_mask(SDG)
    seg_buf = buffer.seg_buf
    seg_grad_buf = buffer.seg_grad_buf
    need_grad = maskg !== zero(SegSeq.ValueMask)
    @inbounds for i in 1:NSeg
        τ = kern.args[i * 5 - 4]
        Ω = kern.args[i * 5 - 3]
        Ω′ = kern.args[i * 5 - 2]
        φ = kern.args[i * 5 - 1]
        δ = kern.args[i * 5]
        if Ω′ == 0
            seg, grad = SegInt.compute_values(τ, Ω, Utils.Zero(), φ, δ,
                                              Val(maskv), Val(maskg))
        else
            seg, grad = SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(maskv), Val(maskg))
        end
        seg_buf[i] = seg
        if need_grad
            grad_buf = seg_grad_buf[i]
            pmask.τ && (grad_buf[1] = grad[1])
            pmask.Ω && (grad_buf[2] = grad[2])
            pmask.Ω′ && (grad_buf[3] = grad[3])
            pmask.φ && (grad_buf[4] = grad[4])
            pmask.ω && (grad_buf[5] = grad[5])
        end
    end
    SegSeq.compute_single_mode!(kern.result, seg_buf, buffer.buffer,
                                need_grad ? seg_grad_buf : nothing,
                                Val(pmask.τ), Val(true))
    return
end

@inline function update!(kern::Kernel{NSeg,T,SDV,SDG,pmask,NArgs},
                         args::NTuple{N}) where {NSeg,T,SDV,SDG,pmask,NArgs,N}
    @assert N == NArgs
    if args == kern.args && kern.evaled
        return
    end
    kern.args = args
    kern.evaled = false
    _update!(kern)
    kern.evaled = true
    return
end

@inline function value_rdis(kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskv = SegSeq.value_mask(SDV)
    @assert maskv.dis
    update!(kern, args)
    return real(kern.result.val.dis)
end

@inline function grad_rdis(g, kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskg = SegSeq.value_mask(SDG)
    @assert maskg.dis
    update!(kern, args)
    grad = kern.result.grad.values
    @inbounds for i in 1:NSeg * 5
        g[i] = real(grad[i].dis)
    end
    return
end

@inline function value_idis(kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskv = SegSeq.value_mask(SDV)
    @assert maskv.dis
    update!(kern, args)
    return imag(kern.result.val.dis)
end

@inline function grad_idis(g, kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskg = SegSeq.value_mask(SDG)
    @assert maskg.dis
    update!(kern, args)
    grad = kern.result.grad.values
    @inbounds for i in 1:NSeg * 5
        g[i] = imag(grad[i].dis)
    end
    return
end

@inline function value_area(kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskv = SegSeq.value_mask(SDV)
    @assert maskv.area
    update!(kern, args)
    return kern.result.val.area
end

@inline function grad_area(g, kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskg = SegSeq.value_mask(SDG)
    @assert maskg.area
    update!(kern, args)
    grad = kern.result.grad.values
    @inbounds for i in 1:NSeg * 5
        g[i] = grad[i].area
    end
    return
end

@inline function value_rcumdis(kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskv = SegSeq.value_mask(SDV)
    @assert maskv.cumdis
    update!(kern, args)
    return real(kern.result.val.cumdis)
end

@inline function grad_rcumdis(g, kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskg = SegSeq.value_mask(SDG)
    @assert maskg.cumdis
    update!(kern, args)
    grad = kern.result.grad.values
    @inbounds for i in 1:NSeg * 5
        g[i] = real(grad[i].cumdis)
    end
    return
end

@inline function value_icumdis(kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskv = SegSeq.value_mask(SDV)
    @assert maskv.cumdis
    update!(kern, args)
    return imag(kern.result.val.cumdis)
end

@inline function grad_icumdis(g, kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskg = SegSeq.value_mask(SDG)
    @assert maskg.cumdis
    update!(kern, args)
    grad = kern.result.grad.values
    @inbounds for i in 1:NSeg * 5
        g[i] = imag(grad[i].cumdis)
    end
    return
end

@inline function value_rdisδ(kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskv = SegSeq.value_mask(SDV)
    @assert maskv.disδ
    update!(kern, args)
    return real(kern.result.val.disδ)
end

@inline function grad_rdisδ(g, kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskg = SegSeq.value_mask(SDG)
    @assert maskg.disδ
    update!(kern, args)
    grad = kern.result.grad.values
    @inbounds for i in 1:NSeg * 5
        g[i] = real(grad[i].disδ)
    end
    return
end

@inline function value_idisδ(kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskv = SegSeq.value_mask(SDV)
    @assert maskv.disδ
    update!(kern, args)
    return imag(kern.result.val.disδ)
end

@inline function grad_idisδ(g, kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskg = SegSeq.value_mask(SDG)
    @assert maskg.disδ
    update!(kern, args)
    grad = kern.result.grad.values
    @inbounds for i in 1:NSeg * 5
        g[i] = imag(grad[i].disδ)
    end
    return
end

@inline function value_areaδ(kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskv = SegSeq.value_mask(SDV)
    @assert maskv.areaδ
    update!(kern, args)
    return kern.result.val.areaδ
end

@inline function grad_areaδ(g, kern::Kernel{NSeg,T,SDV,SDG}, args...) where {NSeg,T,SDV,SDG}
    maskg = SegSeq.value_mask(SDG)
    @assert maskg.areaδ
    update!(kern, args)
    grad = kern.result.grad.values
    @inbounds for i in 1:NSeg * 5
        g[i] = grad[i].areaδ
    end
    return
end

end
