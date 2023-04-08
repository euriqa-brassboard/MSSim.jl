#!/usr/bin/julia

module SymLinear

# Integral for pulses that are piecewise linear in both amplitude and phase.

module SegInt

import ...Utils
import ...SegSeq

using ForwardDiff
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
        SG = SegSeq.SegGrad{A,CD,AG}
        grads = SA[SG(area_grad[1], cumdis_grad[1], area_mode_grad[1]),
                   SG(area_grad[2], cumdis_grad[2], area_mode_grad[2]),
                   SG(area_grad[3], cumdis_grad[3], area_mode_grad[3]),
                   SG(area_grad[4], cumdis_grad[4], area_mode_grad[4]),
                   SG(area_grad[5], cumdis_grad[5], area_mode_grad[5])]
        return res, grads
    end
end

function compute_all_gradients(τ, Ω, Ω′, φ, δ)
    sφ, cφ = @inline sincos(φ)
    d = ForwardDiff.Dual(δ * τ, (δ, 0.0, 0.0, 0.0, τ))
    sd, cd = @inline sincos(δ * τ)
    s = ForwardDiff.Dual(sd, (cd * δ, 0.0, 0.0, 0.0, cd * τ))
    c = ForwardDiff.Dual(cd, (-sd * δ, 0.0, 0.0, 0.0, -sd * τ))
    o = ForwardDiff.Dual(Ω * τ, (Ω, τ, 0.0, 0.0, 0.0))
    o′ = ForwardDiff.Dual(Ω′ * τ^2, (2 * Ω′ * τ, 0.0, τ^2, 0.0, 0.0))
    phase0 = complex(ForwardDiff.Dual(cφ, (0.0, 0.0, 0.0, -sφ, 0.0)),
                     ForwardDiff.Dual(sφ, (0.0, 0.0, 0.0, cφ, 0.0)))
    # phase0_τ = phase0 * ForwardDiff.Dual(τ, (1, 0, 0, 0, 0))
    phase0_τ = complex(ForwardDiff.Dual(cφ * τ, (cφ, 0.0, 0.0, -(sφ * τ), 0.0)),
                        ForwardDiff.Dual(sφ * τ, (sφ, 0.0, 0.0, cφ * τ, 0.0)))

    return @inline (phase0 * displacement_kernel(o, o′, d, s, c),
                    phase0_τ * cumulative_displacement_kernel(o, o′, d, s, c),
                    enclosed_area_kernel(o, o′, d, s, c))
end

end

end
