#!/usr/bin/julia

module SegSeq

using MSSim
const PN = MSSim.PureNumeric
const SL = MSSim.SymLinear
const SS = MSSim.SegSeq

using Test

const τs = [0, 1, 5, 10, 20]
const Ωs = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0, 0.001, 0.02, 1, 5, 10, 20]
const Ω′s = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0,
               0.001, 0.02, 1, 5, 10, 20] ./ 10
const φs = range(0, 2π, 17)
const δs = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0,
             0.001, 0.02, 1, 5, 10, 20] ./ 10

const all_params = Iterators.product(τs, Ωs, Ω′s, φs, δs)

@testset "Single Segment" begin
    T = Float64
    CT = Complex{T}
    A = SS.AreaData{T}
    CD = SS.CumDisData{T,CT}
    AG = SS.AreaModeData{T,CT}

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SeqResultData{T,A,CD,AG}(1, 0)

    for (τ, Ω, Ω′, φ, δ) in all_params
        d = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(true), Val(true), Val(false))
        SS.compute_sequence!(result, [d], buffer)
        @test result.τ == τ
        @test result.area.dis == d.area.dis
        @test result.area.area == d.area.area
        @test result.cumdis.cumdis == d.cumdis.cumdis
        @test result.area_mode.disδ == d.area_mode.disδ
        @test result.area_mode.areaδ == d.area_mode.areaδ
    end
end

@testset "Trivial Segment" begin
    T = Float64
    CT = Complex{T}
    A = SS.AreaData{T}
    CD = SS.CumDisData{T,CT}
    AG = SS.AreaModeData{T,CT}

    buffer = SS.SeqComputeBuffer{T}()
    result2 = SS.SeqResultData{T,A,CD,AG}(2, 0)
    result3 = SS.SeqResultData{T,A,CD,AG}(3, 0)
    result4 = SS.SeqResultData{T,A,CD,AG}(4, 0)

    for (τ, Ω, Ω′, φ, δ) in all_params
        d = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(true), Val(true), Val(false))
        nd = SL.SegInt.compute_values(τ, Ω + Ω′ * τ, -Ω′, φ + δ * τ + π, -δ,
                                      Val(true), Val(true), Val(false))
        @test nd.τ == d.τ
        @test nd.area.dis ≈ -d.area.dis
        @test nd.area.area ≈ -d.area.area
        @test nd.cumdis.cumdis ≈ d.cumdis.cumdis - d.area.dis * τ
        @test nd.area_mode.disδ ≈ d.area_mode.disδ - im * τ * d.area.dis
        @test nd.area_mode.areaδ ≈ d.area_mode.areaδ
        SS.compute_sequence!(result2, [d, nd], buffer)
        @test result2.τ ≈ 2 * τ
        @test result2.area.dis ≈ 0 atol=1e-8
        @test result2.area.area ≈ 0 atol=1e-8
        @test result2.cumdis.cumdis ≈ 2 * d.cumdis.cumdis
        @test result2.area_mode.disδ ≈ 2 * d.area_mode.disδ - 2 * im * τ * d.area.dis
        @test result2.area_mode.areaδ ≈
            (2 * imag(d.area_mode.disδ * conj(d.area.dis))
             - 2 * τ * abs2(d.area.dis) + 2 * d.area_mode.areaδ)

        SS.compute_sequence!(result2, [d, d], buffer)
        @test result2.τ ≈ 2 * τ
        @test result2.area.dis ≈ 2 * d.area.dis atol=1e-8
        @test result2.area.area ≈ 2 * d.area.area atol=1e-8
        @test result2.cumdis.cumdis ≈ 2 * d.cumdis.cumdis + τ * d.area.dis
        @test result2.area_mode.disδ ≈ 2 * d.area_mode.disδ + im * τ * d.area.dis
        @test result2.area_mode.areaδ ≈ τ * abs2(d.area.dis) + 2 * d.area_mode.areaδ

        for τ′ in τs
            d0 = SL.SegInt.compute_values(τ′, 0.0, 0.0, 0.0, 0.0,
                                          Val(true), Val(true), Val(false))
            SS.compute_sequence!(result2, [d0, d], buffer)
            @test result2.τ ≈ τ + τ′
            @test result2.area.dis == d.area.dis
            @test result2.area.area == d.area.area
            @test result2.cumdis.cumdis == d.cumdis.cumdis
            @test result2.area_mode.disδ ≈ d.area_mode.disδ + im * τ′ * d.area.dis
            @test result2.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result2, [d, d0], buffer)
            @test result2.τ ≈ τ + τ′
            @test result2.area.dis == d.area.dis
            @test result2.area.area == d.area.area
            @test result2.cumdis.cumdis ≈ d.cumdis.cumdis + τ′ * d.area.dis
            @test result2.area_mode.disδ == d.area_mode.disδ
            @test result2.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result3, [d0, d0, d], buffer)
            @test result3.τ ≈ τ + τ′ * 2
            @test result3.area.dis == d.area.dis
            @test result3.area.area == d.area.area
            @test result3.cumdis.cumdis == d.cumdis.cumdis
            @test result3.area_mode.disδ ≈ d.area_mode.disδ + 2 * im * τ′ * d.area.dis
            @test result3.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result3, [d0, d, d0], buffer)
            @test result3.τ ≈ τ + τ′ * 2
            @test result3.area.dis == d.area.dis
            @test result3.area.area == d.area.area
            @test result3.cumdis.cumdis ≈ d.cumdis.cumdis + τ′ * d.area.dis
            @test result3.area_mode.disδ ≈ d.area_mode.disδ + im * τ′ * d.area.dis
            @test result3.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result3, [d, d0, d0], buffer)
            @test result3.τ ≈ τ + τ′ * 2
            @test result3.area.dis == d.area.dis
            @test result3.area.area == d.area.area
            @test result3.cumdis.cumdis ≈ d.cumdis.cumdis + 2 * τ′ * d.area.dis
            @test result3.area_mode.disδ == d.area_mode.disδ
            @test result3.area_mode.areaδ == d.area_mode.areaδ
        end
    end
end

end
