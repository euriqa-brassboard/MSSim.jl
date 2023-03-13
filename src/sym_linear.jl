#!/usr/bin/julia

module SymLinear

# Integral for pulses that are piecewise linear in both amplitude and phase.

module SegInt
# Integral for each segments

function displacement(τ, Ω, Ω′, φ, δ)
    phase0 = exp(im * φ)
    phase1 = exp(im * δ * τ)
    return phase0 / δ^2 * ((-im * Ω * δ + Ω′) * (phase1 - 1)
                            - im * Ω′ * δ * τ * phase1)
end

function cumulative_displacement(τ, Ω, Ω′, φ, δ)
    phase0 = exp(im * φ)
    phase1 = exp(im * δ * τ)
    return -phase0 / δ^2 * ((Ω + 2 * im * Ω′ / δ) * (phase1 - 1)
                            + (Ω′ + Ω′ * phase1 - im * Ω * δ) * τ)
end

end

end
