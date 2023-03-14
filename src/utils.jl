#!/usr/bin/julia

module Utils

# Math utility functions
# Note that the sinc and cosc functions below are different from the Base julia version
# in that the input is directly the angle given to the sin and cos functions.

fastabs(x::Number) = abs(x)
fastabs(z::Complex) = abs(real(z)) + abs(imag(z))

const _threshold = 0.01

macro _combined_func(name)
    quote
        @inline $(esc(name))(x::T, s, c) where T =
            fastabs(x) <= _threshold ?
            $(esc(Symbol("_$(name)_small")))(float(x)) :
            $(esc(Symbol("_$(name)_big")))(float(x), s, c)
    end
end

# sin(x) / x
@inline _sinc_small(x::T) where T =
    evalpoly(x^2, T(1), T(-1 / 6), T(1 / 120), T(-1 / 5_040),
             T(1 / 362_880), T(-1 / 39_916_800), T(1 / 6_227_020_800))
@inline _sinc_big(x, s, c) = s / x
@_combined_func sinc

# (x * cos(x) - sin(x)) / x^2
@inline _cosc_small(x::T) where T =
    x * evalpoly(x^2, T(-1 / 3), T(1 / 30), T(-1 / 840), T(1 / 45_360),
                 T(-1 / 3_991_680), T(1 / 518_918_400), T(-1 / 93_405_312_000))
@inline _cosc_big(x, s, c) = (x * c - s) / x^2
@_combined_func cosc

# (1 - cos(x)) / x^2
@inline _cos_f1_small(x::T) where T =
    evalpoly(x^2, T(1 / 2), T(-1 / 24), T(1 / 720), T(-1 / 40_320),
             T(1 / 3_628_800), T(-1 / 479_001_600), T(1 / 87_178_291_200))
@inline _cos_f1_big(x, s, c) = (1 - c) / x^2
@_combined_func cos_f1

# (x - sin(x)) / x^2
@inline _sin_f1_small(x::T) where T =
    x * evalpoly(x^2, T(1 / 6), T(-1 / 120), T(1 / 5_040), T(-1 / 362_880),
                 T(1 / 39_916_800), T(-1 / 6_227_020_800), T(1 / 1_307_670_368_000))
@inline _sin_f1_big(x, s, c) = (x - s) / x^2
@_combined_func sin_f1

# (2 * cos(x) - 2 + x * sin(x)) / x^3
@inline _cos_f2_small(x::T) where T =
    x * evalpoly(x^2, T(-1 / 12), T(1 / 180), T(-1 / 6_720), T(1 / 453_600),
                 T(-1 / 47_900_160), T(1 / 7_264_857_600), T(1 / 1_494_484_992_000))
@inline _cos_f2_big(x, s, c) = (2 * c - 2 + x * s) / x^3
@_combined_func cos_f2

# (2 * sin(x) - x * cos(x) - x) / x^3
@inline _sin_f2_small(x::T) where T =
    evalpoly(x^2, T(1 / 6), T(-1 / 40), T(1 / 1008), T(-1 / 51_840),
             T(1 / 4_435_200), T(-1 / 566_092_800), T(1 / 100_590_336_000))
@inline _sin_f2_big(x, s, c) = (2 * s - x * c - x) / x^3
@_combined_func sin_f2

end
