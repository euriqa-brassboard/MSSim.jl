#!/usr/bin/julia

module Utils

# Math utility functions

@inline mulim(x) = @inline complex(-imag(x), real(x))

fastabs(x::Number) = abs(x)
fastabs(z::Complex) = abs(real(z)) + abs(imag(z))

macro _combined_func(name, threshold)
    quote
        @inline $(esc(name))(x::T, s, c) where T =
            fastabs(x) <= $threshold ?
            $(esc(Symbol("_$(name)_small")))(float(x)) :
            $(esc(Symbol("_$(name)_big")))(float(x), s, c)
    end
end

# sin(x) / x
@inline _sin_c1_small(x::T) where T =
    @inline evalpoly(x^2, (T(1), -1 / T(6), 1 / T(120), -1 / T(5_040),
                           1 / T(362_880), -1 / T(39_916_800), 1 / T(6_227_020_800)))
@inline _sin_c1_big(x, s, c) = s / x
@_combined_func sin_c1 0.3

# (sin(x) - x * cos(x)) / x^2
@inline _sin_c2_small(x::T) where T =
    x * @inline evalpoly(x^2, (1 / T(3), -1 / T(30), 1 / T(840), -1 / T(45_360),
                               1 / T(3_991_680), -1 / T(518_918_400),
                               1 / T(93_405_312_000)))
@inline _sin_c2_big(x, s, c) = (s - c * x) / x^2
@_combined_func sin_c2 0.5

# ((x^2 - 2) * sin(x) + 2 * x * cos(x)) / x^3
@inline _sin_c3_small(x::T) where T =
    @inline evalpoly(x^2, (1 / T(3), -1 / T(10), 1 / T(168), -1 / T(6_480),
                           1 / T(443_520), -1 / T(47_174_400),
                           1 / T(7_185_024_000), -1 / T(1_482_030_950_400)))
@inline _sin_c3_big(x, s, c) = 2 * (x * c - s) / x^3 + s / x
@_combined_func sin_c3 0.7

# (1 - cos(x)) / x^2
@inline _cos_f1_small(x::T) where T =
    @inline evalpoly(x^2, (1 / T(2), -1 / T(24), 1 / T(720), -1 / T(40_320),
                           1 / T(3_628_800), -1 / T(479_001_600),
                           1 / T(87_178_291_200)))
@inline _cos_f1_big(x, s, c) = (1 - c) / x^2
@_combined_func cos_f1 0.45

# (x - sin(x)) / x^2
@inline _sin_f1_small(x::T) where T =
    x * @inline evalpoly(x^2, (1 / T(6), -1 / T(120), 1 / T(5_040), -1 / T(362_880),
                               1 / T(39_916_800), -1 / T(6_227_020_800)))
@inline _sin_f1_big(x, s, c) = (x - s) / x^2
@_combined_func sin_f1 0.3

# (2 - 2 * cos(x) - x * sin(x)) / x^3
@inline _cos_f2_small(x::T) where T =
    x * @inline evalpoly(x^2, (1 / T(12), -1 / T(180), 1 / T(6_720), -1 / T(453_600),
                               1 / T(47_900_160), -1 / T(7_264_857_600),
                               1 / T(1_494_484_992_000)))
@inline _cos_f2_big(x, s, c) = (2 * (1 - c) - s * x) / x^3
@_combined_func cos_f2 0.55

# (2 * sin(x) - x * cos(x) - x) / x^3
@inline _sin_f2_small(x::T) where T =
    @inline evalpoly(x^2, (1 / T(6), -1 / T(40), 1 / T(1_008), -1 / T(51_840),
                           1 / T(4_435_200), -1 / T(566_092_800),
                           1 / T(100_590_336_000), -1 / T(23_712_495_206_400)))
@inline _sin_f2_big(x, s, c) = ((s - x) + (s - x * c)) / x^3
@_combined_func sin_f2 0.7

# (x^2/2 + 1 - cos(x) - x * sin(x)) / x^4
@inline _cos_f3_small(x::T) where T =
    @inline evalpoly(x^2, (1 / T(8), -1 / T(144), 1 / T(5_760), -1 / T(403_200),
                           1 / T(43_545_600), -1 / T(6_706_022_400),
                           1 / T(1_394_852_659_200)))
@inline _cos_f3_big(x, s, c) = (x * (x / 2 - s) + (1 - c)) / (x^2)^2
@_combined_func cos_f3 0.6

# (x^3/3 - sin(x) + x * cos(x)) / x^4
@inline _sin_f3_small(x::T) where T =
    x * @inline evalpoly(x^2, (1 / T(30), -1 / T(840), 1 / T(45_360),
                               -1 / T(3_991_680), 1 / T(518_918_400),
                               -1 / T(93_405_312_000), 1 / T(22_230_464_256_000)))
@inline _sin_f3_big(x, s, c) = (x^3 / 3 + (x * c - s)) / (x^2)^2
@_combined_func sin_f3 0.7

# (-x^3/3 + (4 - x^2) * sin(x) - 4 * x * cos(x)) / x^5
@inline _sin_f4_small(x::T) where T =
    @inline evalpoly(x^2, (1 / T(30), -1 / T(280), 1 / T(9_072),
                           -1 / T(570_240), 1 / T(57_657_600),
                           -1 / T(8_491_392_000), 1 / T(1_710_035_712_000),
                           -1 / T(450_537_408_921_600)))
@inline _sin_f4_big(x, s, c) = (-x^2 * s + 4 * (s - x * c) - x^3 / 3) / (x^2) / (x^3)
@_combined_func sin_f4 1.1

# (-2/3 x^3 + (20 - 7x^2) * sin(x) + (x^2 - 20) * x * cos(x)) / x^6
@inline _sin_f5_small(x::T) where T =
    x * @inline evalpoly(x^2, (1 / T(140), -1 / T(2_268), 1 / T(95_040),
                               -1 / T(7_207_200), 1 / T(849_139_200),
                               -1 / T(142_502_976_000), 1 / T(32_181_243_494_400),
                               -1 / T(9_391_717_310_976_000)))
@inline _sin_f5_big(x::T, s, c) where T =
    (x^3 * (c - T(2) / 3) + (20 * (s - x * c) - 7 * x^2 * s)) / (x^3)^2
@_combined_func sin_f5 1.3

end
