#!/usr/bin/julia

module SymLinear

# Integral for pulses that are piecewise linear in both amplitude and phase.

module SegInt

import ...Utils
using ForwardDiff

# Integral for each segments
@inline function displacement_kernel(o, o′, d, s, c)
    S_C1 = Utils.sin_c1(d, s, c)
    S_C2 = Utils.sin_c2(d, s, c)
    C1 = Utils.cos_f1(d, s, c)
    return complex((o + o′) * S_C1 - o′ * C1, o * d * C1 + o′ * S_C2)
end

@inline function displacement_δ_kernel(o, o′, d, s, c)
    C1 = Utils.cos_f1(d, s, c)
    C2 = Utils.cos_f2(d, s, c)
    S_C2 = Utils.sin_c2(d, s, c)
    S_C3 = Utils.sin_c3(d, s, c)
    return complex(o′ * C2 - (o + o′) * S_C2,
                   o * C1 - o * d * C2 + o′ * S_C3)
end

@inline function cumulative_displacement_kernel(o, o′, d, s, c)
    C1 = Utils.cos_f1(d, s, c)
    S1 = Utils.sin_f1(d, s, c)
    C2 = Utils.cos_f2(d, s, c)
    S2 = Utils.sin_f2(d, s, c)
    return complex(o * C1 + o′ * S2, o * S1 + o′ * C2)
end

# Twice the enclosed area
@inline function enclosed_area_complex_kernel(o, o′, d, s, c)
    a1 = o * (o + o′)
    a2 = o′^2
    C1 = Utils.cos_f1(d, s, c)
    S1 = Utils.sin_f1(d, s, c)
    C3 = Utils.cos_f3(d, s, c)
    S3 = Utils.sin_f3(d, s, c)
    return complex(a1 * C1 + a2 * C3, a1 * S1 + a2 * S3)
end

# Twice the enclosed area
@inline function enclosed_area_kernel(o, o′, d, s, c)
    a1 = o * (o + o′)
    a2 = o′^2
    S1 = Utils.sin_f1(d, s, c)
    S3 = Utils.sin_f3(d, s, c)
    return a1 * S1 + a2 * S3
end

@inline function enclosed_area_δ_kernel(o, o′, d, s, c)
    a1 = o * (o + o′)
    a2 = o′^2
    S2 = Utils.sin_f2(d, s, c)
    S4 = Utils.sin_f4(d, s, c)
    return a1 * S2 + a2 * S4
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

function cumulative_displacement(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    return phase0 * τ * cumulative_displacement_kernel(o, o′, d, s, c)
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

# The values we may care about in each segments
# * Displacement (dis)
# * Gradient of displacement w.r.t. detuning (disδ)
# * Cumulative displacement (cumdis)
# * Enclosed area (area)
# * Gradient of enclosed area w.r.t. detuning (areaδ)
# As well as the gradient of everything above w.r.t. each of the input parameters

@inline function compute_dis_area(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = @inline sincos(d)
    sφ, cφ = @inline sincos(φ)
    phase0 = complex(cφ, sφ)
    return @inline (phase0 * displacement_kernel(o, o′, d, s, c),
                    enclosed_area_kernel(o, o′, d, s, c))
end

@inline function compute_dis_cumdis_area(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = @inline sincos(d)
    sφ, cφ = @inline sincos(φ)
    phase0 = complex(cφ, sφ)
    return @inline (phase0 * displacement_kernel(o, o′, d, s, c),
                    phase0 * τ * cumulative_displacement_kernel(o, o′, d, s, c),
                    enclosed_area_kernel(o, o′, d, s, c))
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
