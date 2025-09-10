#!/usr/bin/julia

module MSSimPureNumericExt

using QuadGK

import MSSim.PureNumeric: displacement, cumulative_displacement,
    enclosed_area_complex, enclosed_area

# Some useful integral for simulating MS gates.
# Done completely numerically. This is likely the slowest
# but should be useful to test other faster implementations.

function displacement(t0, t1, Ω, θ; atol=1e-8, rtol=1e-8)
    f(t) = Ω(t) * cis(θ(t))
    res, err = quadgk(f, t0, t1; atol=atol, rtol=rtol)
    return res
end

function cumulative_displacement(t0, t1, Ω, θ; atol=1e-8, rtol=1e-8)
    f(t) = (t1 - t) * Ω(t) * cis(θ(t))
    res, err = quadgk(f, t0, t1; atol=atol, rtol=rtol)
    return res
end

# Twice the enclosed area
function enclosed_area_complex(t0, t1, Ω, θ; atol=1e-8, rtol=1e-8)
    f(t) = Ω(t) * cis(θ(t)) * displacement(t0, t, Ω, t->-θ(t);
                                             atol=0, rtol=rtol / 10000)
    res, err = quadgk(f, t0, t1; atol=atol, rtol=rtol)
    return res
end

# Twice the enclosed area
function enclosed_area(t0, t1, Ω, θ; atol=1e-8, rtol=1e-8)
    # Only getting the interesting part of the integral.
    f(t) = Ω(t) * imag(cis(θ(t)) * displacement(t0, t, Ω, t->-θ(t),
                                                  atol=0, rtol=rtol / 10000))
    res, err = quadgk(f, t0, t1; atol=atol, rtol=rtol)
    return res
end

end
