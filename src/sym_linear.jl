#!/usr/bin/julia

module SymLinear

# Integral for pulses that are piecewise linear in both amplitude and phase.

module SegInt

import ...Utils
import ...SegSeq

using StaticArrays

# Integral for each segments
@inline function displacement_kernel(o, o′, d, s, c)
    S_C1 = Utils.sin_c1(d, s, c)
    S_C2 = Utils.sin_c2(d, s, c)
    C1 = Utils.cos_f1(d, s, c)
    return complex(muladd(o + o′, S_C1, -o′ * C1),
                   muladd(o * d, C1, o′ * S_C2))
end

@inline function displacement_δ_kernel(o, o′, d, s, c)
    C1 = Utils.cos_f1(d, s, c)
    C2 = Utils.cos_f2(d, s, c)
    S_C2 = Utils.sin_c2(d, s, c)
    S_C3 = Utils.sin_c3(d, s, c)
    return complex(muladd(o + o′, -S_C2, o′ * C2),
                   muladd(o, muladd(-d, C2, C1), o′ * S_C3))
end

@inline function displacement_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ)
    C1 = Utils.cos_f1(d, s, c)
    S_C1 = Utils.sin_c1(d, s, c)
    S_C2 = Utils.sin_c2(d, s, c)
    return (complex(muladd(Ω′, τ, Ω) * c, muladd(Ω′, τ, Ω) * s),
            τ * complex(S_C1, d * C1),
            τ^2 * complex(S_C1 - C1, S_C2))
end

@inline function displacement_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ)
    C1 = Utils.cos_f1(d, s, c)
    C2 = Utils.cos_f2(d, s, c)
    C3_2 = Utils.cos_f3_2(d, s, c)
    S3_3 = Utils.sin_f3_3(d, s, c)
    S_C2 = Utils.sin_c2(d, s, c)
    S_C3 = Utils.sin_c3(d, s, c)

    return ((o + o′) * complex(-s, c),
            τ^2 * complex(-S_C2, muladd(d, -C2, C1)),
            τ^2 * τ * complex(C2 - S_C2, S_C3),
            τ^2 * complex(muladd(o + o′, -S_C3, o′ * C3_2),
                           -muladd(o′, S3_3, o * muladd(d, C3_2, 2 * C2))))
end

@inline function cumulative_displacement_kernel(o, o′, d, s, c)
    C1 = Utils.cos_f1(d, s, c)
    S1 = Utils.sin_f1(d, s, c)
    C2 = Utils.cos_f2(d, s, c)
    S2 = Utils.sin_f2(d, s, c)
    return complex(muladd(o, C1, o′ * S2), muladd(o, S1 * d, o′ * C2))
end

@inline function cumulative_displacement_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ)
    C1 = Utils.cos_f1(d, s, c)
    S1 = Utils.sin_f1(d, s, c)
    C2 = Utils.cos_f2(d, s, c)
    S2 = Utils.sin_f2(d, s, c)
    S3_2 = Utils.sin_f3_2(d, s, c)
    C3_2 = Utils.cos_f3_2(d, s, c)
    S_C1 = Utils.sin_c1(d, s, c)
    S_C2 = Utils.sin_c2(d, s, c)
    return (complex(muladd(o + o′, S_C1, -(o′ * C1)), muladd(o * d, C1, o′ * S_C2)),
            τ^2 * complex(C1, S1 * d), τ * τ^2 * complex(S2, C2),
            τ^2 * complex(-muladd(o, C2, o′ * S3_2), muladd(o, S2, o′ * C3_2)))
end

# Twice the enclosed area
@inline function enclosed_area_complex_kernel(o, o′, d, s, c)
    a1 = o * (o + o′)
    a2 = o′^2
    C1 = Utils.cos_f1(d, s, c)
    S1 = Utils.sin_f1(d, s, c)
    C3 = Utils.cos_f3(d, s, c)
    S3 = Utils.sin_f3(d, s, c)
    return complex(muladd(a1, C1, a2 * C3), muladd(a1, S1 * d, a2 * S3))
end

# Twice the enclosed area
@inline function enclosed_area_kernel(o, o′, d, s, c)
    a1 = o * (o + o′)
    a2 = o′^2
    S1 = Utils.sin_f1(d, s, c)
    S3 = Utils.sin_f3(d, s, c)
    return muladd(a1, S1 * d, a2 * S3)
end

@inline function enclosed_area_δ_kernel(o, o′, d, s, c)
    a1 = o * (o + o′)
    a2 = o′^2
    S2 = Utils.sin_f2(d, s, c)
    S4 = Utils.sin_f4(d, s, c)
    return muladd(a1, S2, a2 * S4)
end

@inline function enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ)
    C1 = Utils.cos_f1(d, s, c)
    S1 = Utils.sin_f1(d, s, c)
    S3 = Utils.sin_f3(d, s, c)
    return (muladd(Ω′, τ, Ω) * d * muladd(o, C1, o′ * S1),
            τ * muladd(2, o, o′) * (S1 * d),
            τ^2 * muladd(2 * o′, S3, o * (S1 * d)))
end

@inline function enclosed_area_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ)
    C1 = Utils.cos_f1(d, s, c)
    S1 = Utils.sin_f1(d, s, c)
    S2 = Utils.sin_f2(d, s, c)
    S3_2 = Utils.sin_f3_2(d, s, c)
    S4 = Utils.sin_f4(d, s, c)
    S5 = Utils.sin_f5(d, s, c)
    S_C1 = Utils.sin_c1(d, s, c)

    return (muladd(muladd(muladd(-2, S1, S_C1), o′, o * (S_C1 - C1)), o, o′^2 * S2),
            τ^2 * muladd(2, o, o′) * S2,
            τ^2 * τ * muladd(2 * o′, S4, o * S2),
            τ^2 * muladd(o * (o + o′), -S3_2, -o′^2 * S5))
end

function displacement(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    return phase0 * displacement_kernel(o, o′, d, s, c)
end

function displacement_δ(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    return phase0 * τ * displacement_δ_kernel(o, o′, d, s, c)
end

function displacement_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)

    τΩs = displacement_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ)
    return (phase0 * τΩs[1], phase0 * τΩs[2], phase0 * τΩs[3],
            Utils.mulim(phase0 * displacement_kernel(o, o′, d, s, c)),
            phase0 * τ * displacement_δ_kernel(o, o′, d, s, c))
end

function displacement_δ_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)

    τΩsδ = displacement_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ)
    return (phase0 * τΩsδ[1], phase0 * τΩsδ[2], phase0 * τΩsδ[3],
            Utils.mulim(phase0 * τ * displacement_δ_kernel(o, o′, d, s, c)),
            phase0 * τΩsδ[4])
end

function cumulative_displacement(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    return phase0 * τ * cumulative_displacement_kernel(o, o′, d, s, c)
end

function cumulative_displacement_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)

    τΩsδ = cumulative_displacement_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ)
    return (phase0 * τΩsδ[1], phase0 * τΩsδ[2], phase0 * τΩsδ[3],
            Utils.mulim(phase0 * τ * cumulative_displacement_kernel(o, o′, d, s, c)),
            phase0 * τΩsδ[4])
end

# Twice the enclosed area
function enclosed_area_complex(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    return enclosed_area_complex_kernel(o, o′, d, s, c)
end

# Twice the enclosed area
function enclosed_area(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    return enclosed_area_kernel(o, o′, d, s, c)
end

function enclosed_area_δ(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    return τ * enclosed_area_δ_kernel(o, o′, d, s, c)
end

function enclosed_area_gradients(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)

    τΩs = enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ)
    return (τΩs[1], τΩs[2], τΩs[3], zero(φ),
            τ * enclosed_area_δ_kernel(o, o′, d, s, c))
end

function enclosed_area_δ_gradients(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)

    τΩsδ = enclosed_area_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ)
    return (τΩsδ[1], τΩsδ[2], τΩsδ[3], zero(φ), τΩsδ[4])
end

# The values we may care about in each segments
# * Displacement (dis)
# * Gradient of displacement w.r.t. detuning (disδ)
# * Cumulative displacement (cumdis)
# * Enclosed area (area)
# * Gradient of enclosed area w.r.t. detuning (areaδ)
# As well as the gradient of everything above w.r.t. each of the input parameters

@inline function (compute_values(τ::_T, Ω, Ω′, φ, δ, ::Val{include_cumdis},
                                 ::Val{include_area_mode}, ::Val{include_grad})
                  where {_T,include_cumdis,include_area_mode,include_grad})

    T = float(_T)
    CT = Complex{T}
    A = SegSeq.AreaData{T}
    CD = include_cumdis ? SegSeq.CumDisData{T,CT} : SegSeq.DummyCumDisData
    AG = include_area_mode ? SegSeq.AreaModeData{T,CT} : SegSeq.DummyAreaModeData

    @inline begin
        d = δ * τ
        o = Ω * τ
        o′ = Ω′ * τ^2
        s, c = sincos(d)
        sφ, cφ = sincos(φ)
        phase0 = complex(cφ, sφ)
        phase0_τ = phase0 * τ

        area = A(phase0 * displacement_kernel(o, o′, d, s, c),
                 enclosed_area_kernel(o, o′, d, s, c))
        if SegSeq.is_dummy(CD)
            cumdis = CD(nothing)
        else
            cumdis = CD(phase0_τ * cumulative_displacement_kernel(o, o′, d, s, c))
        end
        if SegSeq.is_dummy(AG)
            area_mode = AG(nothing, nothing)
        else
            area_mode = AG(phase0_τ * displacement_δ_kernel(o, o′, d, s, c),
                           τ * enclosed_area_δ_kernel(o, o′, d, s, c))
        end
        res = SegSeq.SegData{T,A,CD,AG}(τ, area, cumdis, area_mode)
        if !include_grad
            return res, nothing
        end
        if SegSeq.is_dummy(AG)
            disδ = phase0_τ * displacement_δ_kernel(o, o′, d, s, c)
            areaδ = τ * enclosed_area_δ_kernel(o, o′, d, s, c)
        else
            disδ = area_mode.disδ
            areaδ = area_mode.areaδ
        end
        dis_τΩs = displacement_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ)
        area_τΩs = enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ)
        area_grad = SA[A(phase0 * dis_τΩs[1], area_τΩs[1]),
                       A(phase0 * dis_τΩs[2], area_τΩs[2]),
                       A(phase0 * dis_τΩs[3], area_τΩs[3]),
                       A(Utils.mulim(area.dis), zero(T)),
                       A(disδ, areaδ)]
        if SegSeq.is_dummy(CD)
            cumdis_grad = SA[CD(nothing), CD(nothing), CD(nothing),
                             CD(nothing), CD(nothing)]
        else
            cumdis_τΩsδ = cumulative_displacement_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ)
            cumdis_grad = SA[CD(phase0 * cumdis_τΩsδ[1]),
                             CD(phase0 * cumdis_τΩsδ[2]),
                             CD(phase0 * cumdis_τΩsδ[3]),
                             CD(Utils.mulim(cumdis.cumdis)),
                             CD(phase0 * cumdis_τΩsδ[4])]
        end
        if SegSeq.is_dummy(AG)
            area_mode_grad = SA[AG(nothing, nothing), AG(nothing, nothing),
                                AG(nothing, nothing), AG(nothing, nothing),
                                AG(nothing, nothing)]
        else
            disδ_τΩsδ = displacement_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ)
            areaδ_τΩsδ = enclosed_area_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ)
            area_mode_grad = SA[AG(phase0 * disδ_τΩsδ[1], areaδ_τΩsδ[1]),
                                AG(phase0 * disδ_τΩsδ[2], areaδ_τΩsδ[2]),
                                AG(phase0 * disδ_τΩsδ[3], areaδ_τΩsδ[3]),
                                AG(Utils.mulim(area_mode.disδ), zero(T)),
                                AG(phase0 * disδ_τΩsδ[4], areaδ_τΩsδ[4])]
        end
        SD = SegSeq.SegData{T,A,CD,AG}
        grads = SA[SD(1, area_grad[1], cumdis_grad[1], area_mode_grad[1]),
                   SD(0, area_grad[2], cumdis_grad[2], area_mode_grad[2]),
                   SD(0, area_grad[3], cumdis_grad[3], area_mode_grad[3]),
                   SD(0, area_grad[4], cumdis_grad[4], area_mode_grad[4]),
                   SD(0, area_grad[5], cumdis_grad[5], area_mode_grad[5])]
        return res, grads
    end
end

end

import ..Utils
import ..SegSeq

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

mutable struct System{T,A,CD,AG,MR,need_grad}
    const pulses::Vector{Pulse{T}}
    const modes::Vector{Mode{T}}

    cur_mod::Int
    const seg_buf::Vector{SegSeq.SegData{T,A,CD,AG}}
    const seg_grad_buf::Utils.JaggedMatrix{SegSeq.SegData{T,A,CD,AG}}
    const buffer::SegSeq.SeqComputeBuffer{T}
    const single_result::SegSeq.SingleModeResult{T,A,CD,AG}
    const result::MR

    function System{T}(modes::AbstractVector,
                       ::Val{need_cumdis}, ::Val{need_area_mode},
                       ::Val{need_grad}) where {T,need_cumdis,need_area_mode,need_grad}

        CT = Complex{T}
        A = SegSeq.AreaData{T}
        CD = need_cumdis ? SegSeq.CumDisData{T,CT} : SegSeq.DummyCumDisData
        AG = need_area_mode ? SegSeq.AreaModeData{T,CT} : SegSeq.DummyAreaModeData
        SD = SegSeq.SegData{T,A,CD,AG}

        pulses = Pulse{T}[]
        seg_buf = SD[]
        seg_grad_buf = Utils.JaggedMatrix{SD}()
        buffer = SegSeq.SeqComputeBuffer{T}()
        single_result = SegSeq.SingleModeResult{T,A,CD,AG}()
        result = SegSeq.MultiModeResult{T}(Val(need_cumdis), Val(need_area_mode))
        return new{T,A,CD,AG,typeof(result),need_grad}(pulses, modes, 0, seg_buf,
                                                       seg_grad_buf, buffer,
                                                       single_result, result)
    end
    function System{T}(::Val{need_cumdis}, ::Val{need_area_mode},
                       ::Val{need_grad}) where {T,need_cumdis,need_area_mode,need_grad}
        return System{T}(Mode{T}[], Val(need_cumdis),
                         Val(need_area_mode), Val(need_grad))
    end
end

@inline function _fill_seg_buf!(sys::System{T,A,CD,AG,MR,need_grad},
                        mode_idx) where {T,A,CD,AG,MR,need_grad}
    if mode_idx == sys.cur_mod
        return
    end
    need_cumdis = !SegSeq.is_dummy(CD)
    need_area_mode = !SegSeq.is_dummy(AG)

    nseg = length(sys.pulses)
    resize!(sys.seg_buf, nseg)
    empty!(sys.seg_grad_buf)

    mode = sys.modes[mode_idx]
    φ = zero(T)
    Ω = zero(T)
    @inline for i in 1:nseg
        pulse = sys.pulses[i]
        Ω += pulse.dΩ
        φ += pulse.dφ
        δ = pulse.ω - mode.ω
        seg, grad = SegInt.compute_values(pulse.τ, Ω, pulse.Ω′, φ, δ,
                                          Val(need_cumdis), Val(need_area_mode),
                                          Val(need_grad))
        sys.seg_buf[i] = seg
        push!(sys.seg_grad_buf, grad)
        φ += pulse.τ * δ
        Ω += pulse.τ * pulse.Ω′
    end
end

function compute!(sys::System{T,A,CD,AG,MR,need_grad}) where {T,A,CD,AG,MR,need_grad}
    nmodes = length(sys.modes)
    nseg = length(sys.pulses)
    need_cumdis = !SegSeq.is_dummy(CD)
    need_area_mode = !SegSeq.is_dummy(AG)

    SegSeq.init_multi_mode_result!(sys.result, nmodes)

    result = sys.result
    single_result = sys.single_result

    @inline for mode_idx in 1:nmodes
        _fill_seg_buf!(sys, mode_idx)
        mode = sys.modes[mode_idx]
        compute_single_mode!(single_result, sys.seg_buf, sys.buffer,
                             need_grad ? sys.seg_grad_buf : nothing)

        dis_scale = mode.dis_scale
        area_scale = mode.area_scale

        if mode_idx == 1
            result.τ = single_result.τ
        end
        result.dis[mode_idx] = dis_scale * single_result.area.dis
        result.area += area_scale * single_result.area.area
        if need_cumdis
            result.cumdis[mode_idx] = dis_scale * single_result.cumdis.cumdis
        end
        if need_area_mode
            result.disδ[mode_idx] = dis_scale * single_result.area_mode.disδ
            result.areaδ[mode_idx] = area_scale * single_result.area_mode.areaδ
        end
        if need_grad
            if mode_idx == 1
                resize!(result.τ_grad, single_result.τ_grad)
                result.τ_grad.values .= single_result.τ_grad.values

                resize!(result.area_grad, single_result.area_grad)
                result.area_grad.values .= 0
                if need_area_mode
                    resize!(result.areaδ_grad, single_result.area_mode_grad)
                    result.areaδ_grad.values .= 0
                end
            end
            dis_grad = result.dis_grad[mode_idx]
            resize!(dis_grad, single_result.area_grad)
            if need_cumdis
                cumdis_grad = result.cumdis_grad[mode_idx]
                resize!(cumdis_grad, single_result.cumdis_grad)
            end
            if need_area_mode
                disδ_grad = result.disδ_grad[mode_idx]
                resize!(disδ_grad, single_result.area_mode_grad)
                areaδ_grad = result.areaδ_grad[mode_idx]
                resize!(areaδ_grad, single_result.area_mode_grad)
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
                pulse = sys.pulses[seg_idx]
                δ = pulse.ω - mode.ω

                res_dis_sgrad = dis_grad[seg_idx]
                res_area_sgrad = result.area_grad[seg_idx]
                area_sgrad = single_result.area_grad[seg_idx]

                # displacement
                # τ
                res_dis_sgrad[1] = dis_scale * (area_sgrad[1].dis +
                    dis_dΩ * pulse.Ω′ + dis_dφ * δ)
                # Ω
                new_dis_dΩ = area_sgrad[2].dis + dis_dΩ
                res_dis_sgrad[2] = dis_scale * new_dis_dΩ
                # Ω′
                res_dis_sgrad[3] = dis_scale * (area_sgrad[3].dis + dis_dΩ * pulse.τ)
                # φ
                new_dis_dφ = area_sgrad[4].dis + dis_dφ
                res_dis_sgrad[4] = dis_scale * new_dis_dφ
                # ω
                res_dis_sgrad[5] = dis_scale * (area_sgrad[5].dis + dis_dφ * pulse.τ)
                dis_dΩ = new_dis_dΩ
                dis_dφ = new_dis_dφ

                # area
                # τ
                res_area_sgrad[1] += area_scale * (area_sgrad[1].area +
                    area_dΩ * pulse.Ω′ + area_dφ * δ)
                # Ω
                new_area_dΩ = area_sgrad[2].area + area_dΩ
                res_area_sgrad[2] += area_scale * new_area_dΩ
                # Ω′
                res_area_sgrad[3] += area_scale * (area_sgrad[3].area +
                    area_dΩ * pulse.τ)
                # φ
                new_area_dφ = area_sgrad[4].area + area_dφ
                res_area_sgrad[4] += area_scale * new_area_dφ
                # ω
                res_area_sgrad[5] += area_scale * (area_sgrad[5].area +
                    area_dφ * pulse.τ)
                area_dΩ = new_area_dΩ
                area_dφ = new_area_dφ

                if need_cumdis
                    res_cumdis_sgrad = cumdis_grad[seg_idx]
                    cumdis_sgrad = single_result.cumdis_grad[seg_idx]
                    # τ
                    res_cumdis_sgrad[1] = dis_scale * (cumdis_sgrad[1].cumdis +
                        cumdis_dΩ * pulse.Ω′ + cumdis_dφ * δ)
                    # Ω
                    new_cumdis_dΩ = cumdis_sgrad[2].cumdis + cumdis_dΩ
                    res_cumdis_sgrad[2] = dis_scale * new_cumdis_dΩ
                    # Ω′
                    res_cumdis_sgrad[3] = dis_scale * (cumdis_sgrad[3].cumdis +
                        cumdis_dΩ * pulse.τ)
                    # φ
                    new_cumdis_dφ = cumdis_sgrad[4].cumdis + cumdis_dφ
                    res_cumdis_sgrad[4] = dis_scale * new_cumdis_dφ
                    # ω
                    res_cumdis_sgrad[5] = dis_scale * (cumdis_sgrad[5].cumdis +
                        cumdis_dφ * pulse.τ)
                    cumdis_dΩ = new_cumdis_dΩ
                    cumdis_dφ = new_cumdis_dφ
                end

                if need_area_mode
                    res_disδ_sgrad = disδ_grad[seg_idx]
                    res_areaδ_sgrad = areaδ_grad[seg_idx]
                    area_mode_sgrad = single_result.area_mode_grad[seg_idx]

                    # displacement
                    # τ
                    res_disδ_sgrad[1] = dis_scale * (area_mode_sgrad[1].disδ +
                        disδ_dΩ * pulse.Ω′ + disδ_dφ * δ)
                    # Ω
                    new_disδ_dΩ = area_mode_sgrad[2].disδ + disδ_dΩ
                    res_disδ_sgrad[2] = dis_scale * new_disδ_dΩ
                    # Ω′
                    res_disδ_sgrad[3] = dis_scale * (area_mode_sgrad[3].disδ +
                        disδ_dΩ * pulse.τ)
                    # φ
                    new_disδ_dφ = area_mode_sgrad[4].disδ + disδ_dφ
                    res_disδ_sgrad[4] = dis_scale * new_disδ_dφ
                    # ω
                    res_disδ_sgrad[5] = dis_scale * (area_mode_sgrad[5].disδ +
                        disδ_dφ * pulse.τ)
                    disδ_dΩ = new_disδ_dΩ
                    disδ_dφ = new_disδ_dφ

                    # area
                    # τ
                    res_areaδ_sgrad[1] = area_scale * (area_mode_sgrad[1].areaδ +
                        areaδ_dΩ * pulse.Ω′ + areaδ_dφ * δ)
                    # Ω
                    new_areaδ_dΩ = area_mode_sgrad[2].areaδ + areaδ_dΩ
                    res_areaδ_sgrad[2] = area_scale * new_areaδ_dΩ
                    # Ω′
                    res_areaδ_sgrad[3] = area_scale * (area_mode_sgrad[3].areaδ +
                        areaδ_dΩ * pulse.τ)
                    # φ
                    new_areaδ_dφ = area_mode_sgrad[4].areaδ + areaδ_dφ
                    res_areaδ_sgrad[4] = area_scale * new_areaδ_dφ
                    # ω
                    res_areaδ_sgrad[5] = area_scale * (area_mode_sgrad[5].areaδ +
                        areaδ_dφ * pulse.τ)
                    areaδ_dΩ = new_areaδ_dΩ
                    areaδ_dφ = new_areaδ_dφ
                end
            end
        end
    end
    return
end


end
