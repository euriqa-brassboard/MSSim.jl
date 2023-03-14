#!/usr/bin/julia

module SymLinear

# Integral for pulses that are piecewise linear in both amplitude and phase.

module SegInt

import ...Utils

# Integral for each segments
function displacement(τ, Ω, Ω′, φ, δ)
    phase0 = exp(im * φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    S = Utils.sinc(d, s, c)
    C = Utils.cosc(d, s, c)
    C1 = Utils.cos_f1(d, s, c)
    return phase0 * ((im * (o * d) - o′) * C1 + (o + o′) * S - im * (o′ * C))
end

function cumulative_displacement(τ, Ω, Ω′, φ, δ)
    phase0 = exp(im * φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = sincos(d)
    C1 = Utils.cos_f1(d, s, c)
    S1 = Utils.sin_f1(d, s, c)
    C2 = Utils.cos_f2(d, s, c)
    S2 = Utils.sin_f2(d, s, c)
    return phase0 * τ * (o * C1 + o′ * S2 + im * (o * S1 - o′ * C2))
end

function enclosed_area_complex(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2

    a1 = o^2 + o * o′
    a2 = o′^2

    s, c = sincos(d)
    C1 = Utils.cos_f1(d, s, c)
    S1 = Utils.sin_f1(d, s, c)
    C3 = Utils.cos_f3(d, s, c)
    S3 = Utils.sin_f3(d, s, c)
    return a1 * C1 + a2 * C3 + im * (a1 * S1 + a2 * S3)
end

function enclosed_area(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    a1 = o^2 + o * o′
    a2 = o′^2
    s, c = sincos(d)
    C1 = Utils.cos_f1(d, s, c)
    C3 = Utils.cos_f3(d, s, c)
    return a1 * C1 + a2 * C3
end

end

end
