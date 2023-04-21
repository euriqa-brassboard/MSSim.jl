#!/usr/bin/julia

module SymLinear

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
    s, c = sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_c1, sin_c2, cos_f1)
    return phase0 * displacement_kernel(o, o′, d, s, c, V)
end

function displacement_δ(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, cos_f2, sin_c2, sin_c3)
    return phase0 * τ * displacement_δ_kernel(o, o′, d, s, c, V)
end

function displacement_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
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
    s, c = sincos(d)
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
    s, c = sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, cos_f2, sin_f2)

    return phase0 * τ * cumulative_displacement_kernel(o, o′, d, s, c, V)
end

function cumulative_displacement_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
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
    s, c = sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, cos_f3, sin_f3)

    return enclosed_area_complex_kernel(o, o′, d, s, c, V)
end

# Twice the enclosed area
function enclosed_area(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_f1, sin_f3)

    return enclosed_area_kernel(o, o′, d, s, c, V)
end

function enclosed_area_δ(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_f2, sin_f4)

    return τ * enclosed_area_δ_kernel(o, o′, d, s, c, V)
end

function enclosed_area_gradients(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, sin_f2, sin_f3, sin_f4)

    τΩs = enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (τΩs[1], τΩs[2], τΩs[3], zero(φ),
            τ * enclosed_area_δ_kernel(o, o′, d, s, c, V))
end

function enclosed_area_δ_gradients(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
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
@inline function (compute_values(τ::_T, Ω, Ω′, φ, δ, ::Val{need_cumdis},
                                 ::Val{need_area_mode}, ::Val{need_grad})
                  where {_T,need_cumdis,need_area_mode,need_grad})

    T = float(_T)
    CT = Complex{T}
    D = CT
    A = T
    CD = need_cumdis ? CT : Nothing
    AG = need_area_mode ? SegSeq.AreaModeData{T,CT} : SegSeq.DummyAreaModeData
    SD = SegSeq.SegData{T,D,A,CD,AG}

    @inline begin
        d = δ * τ
        o = Ω * τ
        o′ = Ω′ * τ^2
        s, c = sincos(d)
        sφ, cφ = sincos(φ)
        phase0 = complex(cφ, sφ)
        phase0_τ = phase0 * τ
        V = @gen_trig_ratios(d, s, c, sin_c1, sin_c2, sin_c3,
                             cos_f1, sin_f1, cos_f2, sin_f2,
                             cos_f3_2, sin_f3, sin_f3_2, sin_f3_3, sin_f4, sin_f5)

        dis = Utils.mul(phase0, displacement_kernel(o, o′, d, s, c, V))
        area = enclosed_area_kernel(o, o′, d, s, c, V)
        if !need_cumdis
            cumdis = nothing
        else
            cumdis = Utils.mul(phase0_τ, cumulative_displacement_kernel(o, o′, d,
                                                                         s, c, V))
        end
        if !need_area_mode
            area_mode = AG(nothing, nothing)
        else
            area_mode = AG(Utils.mul(phase0_τ, displacement_δ_kernel(o, o′, d,
                                                                       s, c, V)),
                           τ * enclosed_area_δ_kernel(o, o′, d, s, c, V))
        end
        res = SD(τ, dis, area, cumdis, area_mode)
        if !need_grad
            return res, nothing
        end
        if !need_area_mode
            disδ = Utils.mul(phase0_τ, displacement_δ_kernel(o, o′, d, s, c, V))
            areaδ = τ * enclosed_area_δ_kernel(o, o′, d, s, c, V)
        else
            disδ = area_mode.disδ
            areaδ = area_mode.areaδ
        end
        dis_τΩs = displacement_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
        dis_grad = SA[Utils.mul(phase0, dis_τΩs[1]), Utils.mul(phase0, dis_τΩs[2]),
                      Utils.mul(phase0, dis_τΩs[3]), Utils.mulim(dis), disδ]
        area_τΩs = enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
        area_grad = SA[area_τΩs[1], area_τΩs[2], area_τΩs[3], zero(T), areaδ]
        if !need_cumdis
            cumdis_grad = SA[nothing, nothing, nothing, nothing, nothing]
        else
            cumdis_τΩsδ = cumulative_displacement_τΩsδ_kernel(o, o′, d, s, c,
                                                                    Ω, Ω′, τ, V)
            cumdis_grad = SA[Utils.mul(phase0, cumdis_τΩsδ[1]),
                             Utils.mul(phase0, cumdis_τΩsδ[2]),
                             Utils.mul(phase0, cumdis_τΩsδ[3]),
                             Utils.mulim(cumdis),
                             Utils.mul(phase0, cumdis_τΩsδ[4])]
        end
        if !need_area_mode
            area_mode_grad = SA[AG(nothing, nothing), AG(nothing, nothing),
                                AG(nothing, nothing), AG(nothing, nothing),
                                AG(nothing, nothing)]
        else
            disδ_τΩsδ = displacement_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
            areaδ_τΩsδ = enclosed_area_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
            area_mode_grad = SA[AG(Utils.mul(phase0, disδ_τΩsδ[1]), areaδ_τΩsδ[1]),
                                AG(Utils.mul(phase0, disδ_τΩsδ[2]), areaδ_τΩsδ[2]),
                                AG(Utils.mul(phase0, disδ_τΩsδ[3]), areaδ_τΩsδ[3]),
                                AG(Utils.mulim(area_mode.disδ), zero(T)),
                                AG(Utils.mul(phase0, disδ_τΩsδ[4]), areaδ_τΩsδ[4])]
        end
        grads = SA[SD(1, dis_grad[1], area_grad[1], cumdis_grad[1], area_mode_grad[1]),
                   SD(0, dis_grad[2], area_grad[2], cumdis_grad[2], area_mode_grad[2]),
                   SD(0, dis_grad[3], area_grad[3], cumdis_grad[3], area_mode_grad[3]),
                   SD(0, dis_grad[4], area_grad[4], cumdis_grad[4], area_mode_grad[4]),
                   SD(0, dis_grad[5], area_grad[5], cumdis_grad[5], area_mode_grad[5])]
        return res, grads
    end
end

end

import ..Utils
import ..SegSeq

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

mutable struct System{T,D,A,CD,AG,MR,need_grad}
    const pulses::Vector{Pulse{T}}
    const modes::Vector{Mode{T}}

    cur_mod::Int
    const seg_buf::Vector{SegSeq.SegData{T,D,A,CD,AG}}
    const seg_grad_buf::Utils.JaggedMatrix{SegSeq.SegData{T,D,A,CD,AG}}
    const buffer::SegSeq.SeqComputeBuffer{T}
    const single_result::SegSeq.SingleModeResult{T,D,A,CD,AG}
    const result::MR

    function System{T}(modes::AbstractVector,
                       ::Val{need_cumdis}, ::Val{need_area_mode},
                       ::Val{need_grad}) where {T,need_cumdis,need_area_mode,need_grad}

        CT = Complex{T}
        D = CT
        A = T
        CD = need_cumdis ? CT : Nothing
        AG = need_area_mode ? SegSeq.AreaModeData{T,CT} : SegSeq.DummyAreaModeData
        SD = SegSeq.SegData{T,D,A,CD,AG}

        pulses = Pulse{T}[]
        seg_buf = SD[]
        seg_grad_buf = Utils.JaggedMatrix{SD}()
        buffer = SegSeq.SeqComputeBuffer{T}()
        single_result = SegSeq.SingleModeResult{T,D,A,CD,AG}()
        result = SegSeq.MultiModeResult{T}(Val(need_cumdis), Val(need_area_mode))
        return new{T,D,A,CD,AG,typeof(result),need_grad}(pulses, modes, 0, seg_buf,
                                                         seg_grad_buf, buffer,
                                                         single_result, result)
    end
    function System{T}(::Val{need_cumdis}, ::Val{need_area_mode},
                       ::Val{need_grad}) where {T,need_cumdis,need_area_mode,need_grad}
        return System{T}(Mode{T}[], Val(need_cumdis),
                         Val(need_area_mode), Val(need_grad))
    end
end

@inline function _fill_seg_buf!(sys::System{T,D,A,CD,AG,MR,need_grad},
                                mode_idx) where {T,D,A,CD,AG,MR,need_grad}
    need_cumdis = CD !== Nothing
    need_area_mode = !SegSeq.is_dummy(AG)

    nseg = length(sys.pulses)
    resize!(sys.seg_buf, nseg)
    empty!(sys.seg_grad_buf)

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
        seg, grad = SegInt.compute_values(pulse.τ, Ω, pulse.Ω′, φ, δ,
                                          Val(need_cumdis), Val(need_area_mode),
                                          Val(need_grad))
        seg_buf[i] = seg
        if need_grad
            push!(seg_grad_buf, grad)
        end
        φ = muladd(pulse.τ, δ, φ)
        Ω = muladd(pulse.τ, pulse.Ω′, Ω)
    end
end

function compute!(sys::System{T,D,A,CD,AG,MR,need_grad}) where {T,D,A,CD,AG,MR,need_grad}
    nmodes = length(sys.modes)
    nseg = length(sys.pulses)
    need_cumdis = CD !== Nothing
    need_area_mode = !SegSeq.is_dummy(AG)

    SegSeq.init_multi_mode_result!(sys.result, nmodes, Val(need_grad))

    result = sys.result
    single_result = sys.single_result
    pulses = sys.pulses

    @inline @inbounds for mode_idx in 1:nmodes
        _fill_seg_buf!(sys, mode_idx)
        mode = sys.modes[mode_idx]
        SegSeq.compute_single_mode!(single_result, sys.seg_buf, sys.buffer,
                                    need_grad ? sys.seg_grad_buf : nothing)

        dis_scale = mode.dis_scale
        area_scale = mode.area_scale

        if mode_idx == 1
            result.τ = single_result.val.τ
        end
        result.dis[mode_idx] = dis_scale * single_result.val.dis
        result.area = muladd(area_scale, single_result.val.area, result.area)
        if need_cumdis
            result.cumdis[mode_idx] = dis_scale * single_result.val.cumdis
        end
        if need_area_mode
            result.disδ[mode_idx] = dis_scale * single_result.val.area_mode.disδ
            result.areaδ[mode_idx] = area_scale * single_result.val.area_mode.areaδ
        end
        if need_grad
            if mode_idx == 1
                resize!(result.τ_grad, single_result.grad)
                result.τ_grad.values .= getfield.(single_result.grad.values, :τ)

                resize!(result.area_grad, single_result.grad)
                result.area_grad.values .= 0
            end
            dis_grad = result.dis_grad[mode_idx]
            resize!(dis_grad, single_result.grad)
            if need_cumdis
                cumdis_grad = result.cumdis_grad[mode_idx]
                resize!(cumdis_grad, single_result.grad)
            end
            if need_area_mode
                disδ_grad = result.disδ_grad[mode_idx]
                resize!(disδ_grad, single_result.grad)
                areaδ_grad = result.areaδ_grad[mode_idx]
                resize!(areaδ_grad, single_result.grad)
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

                res_dis_sgrad = dis_grad[seg_idx]
                res_area_sgrad = result.area_grad[seg_idx]
                sgrad = single_result.grad[seg_idx]

                # displacement
                # τ
                res_dis_sgrad[1] = dis_scale * muladd(dis_dφ, δ,
                                                      muladd(dis_dΩ, pulse.Ω′,
                                                             sgrad[1].dis))
                # Ω
                new_dis_dΩ = sgrad[2].dis + dis_dΩ
                res_dis_sgrad[2] = dis_scale * new_dis_dΩ
                # Ω′
                res_dis_sgrad[3] = dis_scale * muladd(dis_dΩ, pulse.τ,
                                                      sgrad[3].dis)
                # φ
                new_dis_dφ = sgrad[4].dis + dis_dφ
                res_dis_sgrad[4] = dis_scale * new_dis_dφ
                # ω
                res_dis_sgrad[5] = dis_scale * muladd(dis_dφ, pulse.τ,
                                                      sgrad[5].dis)
                dis_dΩ = new_dis_dΩ
                dis_dφ = new_dis_dφ

                # area
                # τ
                res_area_sgrad[1] = muladd(muladd(area_dφ, δ,
                                                  muladd(area_dΩ, pulse.Ω′,
                                                         sgrad[1].area)),
                                           area_scale, res_area_sgrad[1])
                # Ω
                new_area_dΩ = sgrad[2].area + area_dΩ
                res_area_sgrad[2] = muladd(area_scale, new_area_dΩ, res_area_sgrad[2])
                # Ω′
                res_area_sgrad[3] = muladd(muladd(area_dΩ, pulse.τ,
                                                  sgrad[3].area),
                                           area_scale, res_area_sgrad[3])
                # φ
                new_area_dφ = sgrad[4].area + area_dφ
                res_area_sgrad[4] = muladd(area_scale, new_area_dφ,
                                           res_area_sgrad[4])
                # ω
                res_area_sgrad[5] = muladd(muladd(area_dφ, pulse.τ,
                                                  sgrad[5].area),
                                           area_scale, res_area_sgrad[5])
                area_dΩ = new_area_dΩ
                area_dφ = new_area_dφ

                if need_cumdis
                    res_cumdis_sgrad = cumdis_grad[seg_idx]
                    # τ
                    res_cumdis_sgrad[1] = dis_scale * muladd(
                        cumdis_dφ, δ,
                        muladd(cumdis_dΩ, pulse.Ω′, sgrad[1].cumdis))
                    # Ω
                    new_cumdis_dΩ = sgrad[2].cumdis + cumdis_dΩ
                    res_cumdis_sgrad[2] = dis_scale * new_cumdis_dΩ
                    # Ω′
                    res_cumdis_sgrad[3] = dis_scale * muladd(cumdis_dΩ, pulse.τ,
                                                             sgrad[3].cumdis)
                    # φ
                    new_cumdis_dφ = sgrad[4].cumdis + cumdis_dφ
                    res_cumdis_sgrad[4] = dis_scale * new_cumdis_dφ
                    # ω
                    res_cumdis_sgrad[5] = dis_scale * muladd(cumdis_dφ, pulse.τ,
                                                             sgrad[5].cumdis)
                    cumdis_dΩ = new_cumdis_dΩ
                    cumdis_dφ = new_cumdis_dφ
                end

                if need_area_mode
                    res_disδ_sgrad = disδ_grad[seg_idx]
                    res_areaδ_sgrad = areaδ_grad[seg_idx]

                    # displacement
                    # τ
                    res_disδ_sgrad[1] = dis_scale * muladd(
                        disδ_dφ, δ, muladd(disδ_dΩ, pulse.Ω′, sgrad[1].area_mode.disδ))
                    # Ω
                    new_disδ_dΩ = sgrad[2].area_mode.disδ + disδ_dΩ
                    res_disδ_sgrad[2] = dis_scale * new_disδ_dΩ
                    # Ω′
                    res_disδ_sgrad[3] = dis_scale * muladd(disδ_dΩ, pulse.τ,
                                                            sgrad[3].area_mode.disδ)
                    # φ
                    new_disδ_dφ = sgrad[4].area_mode.disδ + disδ_dφ
                    res_disδ_sgrad[4] = dis_scale * new_disδ_dφ
                    # ω
                    res_disδ_sgrad[5] = dis_scale * muladd(disδ_dφ, pulse.τ,
                                                            sgrad[5].area_mode.disδ)
                    disδ_dΩ = new_disδ_dΩ
                    disδ_dφ = new_disδ_dφ

                    # area
                    # τ
                    res_areaδ_sgrad[1] = area_scale * muladd(
                        areaδ_dφ, δ,
                        muladd(areaδ_dΩ, pulse.Ω′, sgrad[1].area_mode.areaδ))
                    # Ω
                    new_areaδ_dΩ = sgrad[2].area_mode.areaδ + areaδ_dΩ
                    res_areaδ_sgrad[2] = area_scale * new_areaδ_dΩ
                    # Ω′
                    res_areaδ_sgrad[3] = area_scale * muladd(areaδ_dΩ, pulse.τ,
                                                              sgrad[3].area_mode.areaδ)
                    # φ
                    new_areaδ_dφ = sgrad[4].area_mode.areaδ + areaδ_dφ
                    res_areaδ_sgrad[4] = area_scale * new_areaδ_dφ
                    # ω
                    res_areaδ_sgrad[5] = area_scale * muladd(areaδ_dφ, pulse.τ,
                                                              sgrad[5].area_mode.areaδ)
                    areaδ_dΩ = new_areaδ_dΩ
                    areaδ_dφ = new_areaδ_dφ
                end
            end
        end
    end
    return
end


end
