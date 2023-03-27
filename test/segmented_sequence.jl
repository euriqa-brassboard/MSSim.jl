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
    result = SS.SeqResultData{T,A,CD,AG}()

    for (τ, Ω, Ω′, φ, δ) in all_params
        d = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(true), Val(true), Val(false))
        SS.compute_sequence!(result, [d], buffer)
        @test result.τ == τ
        @test result.area.dis == d.area.dis
        @test result.area.area == d.area.area
        @test result.cumdis.cumdis == d.cumdis.cumdis
        @test result.area_mode.disδ == d.area_mode.disδ
        @test result.area_mode.areaδ == d.area_mode.areaδ

        for (need_cumdis, need_area_mode) in Iterators.product((false, true),
                                                               (false, true))
            CD′ = need_cumdis ? CD : SS.DummyCumDisData
            AG′ = need_area_mode ? AG : SS.DummyAreaModeData

            result′ = SS.SeqResultData{T,A,CD′,AG′}()
            d′ = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(need_cumdis),
                                           Val(need_area_mode), Val(false))
            SS.compute_sequence!(result′, [d′], buffer)
            @test result′.τ == τ
            @test result′.area.dis == d.area.dis
            @test result′.area.area == d.area.area
            if need_cumdis
                @test result′.cumdis.cumdis == d.cumdis.cumdis
            end
            if need_area_mode
                @test result′.area_mode.disδ == d.area_mode.disδ
                @test result′.area_mode.areaδ == d.area_mode.areaδ
            end
        end
    end
end

@testset "Trivial Segment" begin
    T = Float64
    CT = Complex{T}
    A = SS.AreaData{T}
    CD = SS.CumDisData{T,CT}
    AG = SS.AreaModeData{T,CT}

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SeqResultData{T,A,CD,AG}()

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
        SS.compute_sequence!(result, [d, nd], buffer)
        @test result.τ ≈ 2 * τ
        @test result.area.dis ≈ 0 atol=1e-8
        @test result.area.area ≈ 0 atol=1e-8
        @test result.cumdis.cumdis ≈ 2 * d.cumdis.cumdis
        @test result.area_mode.disδ ≈ 2 * d.area_mode.disδ - 2 * im * τ * d.area.dis
        @test result.area_mode.areaδ ≈
            (2 * imag(d.area_mode.disδ * conj(d.area.dis))
             - 2 * τ * abs2(d.area.dis) + 2 * d.area_mode.areaδ)

        SS.compute_sequence!(result, [d, d], buffer)
        @test result.τ ≈ 2 * τ
        @test result.area.dis ≈ 2 * d.area.dis atol=1e-8
        @test result.area.area ≈ 2 * d.area.area atol=1e-8
        @test result.cumdis.cumdis ≈ 2 * d.cumdis.cumdis + τ * d.area.dis
        @test result.area_mode.disδ ≈ 2 * d.area_mode.disδ + im * τ * d.area.dis
        @test result.area_mode.areaδ ≈ τ * abs2(d.area.dis) + 2 * d.area_mode.areaδ

        for (need_cumdis, need_area_mode) in Iterators.product((false, true),
                                                               (false, true))
            CD′ = need_cumdis ? CD : SS.DummyCumDisData
            AG′ = need_area_mode ? AG : SS.DummyAreaModeData

            result′ = SS.SeqResultData{T,A,CD′,AG′}()
            d′ = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(need_cumdis),
                                           Val(need_area_mode), Val(false))
            SS.compute_sequence!(result′, [d′, d′], buffer)
            @test result′.τ == 2 * τ
            @test result′.area.dis ≈ 2 * d.area.dis atol=1e-8
            @test result′.area.area ≈ 2 * d.area.area atol=1e-8
            if need_cumdis
                @test result′.cumdis.cumdis ≈ 2 * d.cumdis.cumdis + τ * d.area.dis
            end
            if need_area_mode
                @test result′.area_mode.disδ ≈
                    2 * d.area_mode.disδ + im * τ * d.area.dis
                @test result′.area_mode.areaδ ≈
                    τ * abs2(d.area.dis) + 2 * d.area_mode.areaδ
            end
        end

        for τ′ in τs
            d0 = SL.SegInt.compute_values(τ′, 0.0, 0.0, 0.0, 0.0,
                                          Val(true), Val(true), Val(false))
            SS.compute_sequence!(result, [d0, d], buffer)
            @test result.τ ≈ τ + τ′
            @test result.area.dis == d.area.dis
            @test result.area.area == d.area.area
            @test result.cumdis.cumdis == d.cumdis.cumdis
            @test result.area_mode.disδ ≈ d.area_mode.disδ + im * τ′ * d.area.dis
            @test result.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result, [d, d0], buffer)
            @test result.τ ≈ τ + τ′
            @test result.area.dis == d.area.dis
            @test result.area.area == d.area.area
            @test result.cumdis.cumdis ≈ d.cumdis.cumdis + τ′ * d.area.dis
            @test result.area_mode.disδ == d.area_mode.disδ
            @test result.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result, [d0, d0, d], buffer)
            @test result.τ ≈ τ + τ′ * 2
            @test result.area.dis == d.area.dis
            @test result.area.area == d.area.area
            @test result.cumdis.cumdis == d.cumdis.cumdis
            @test result.area_mode.disδ ≈ d.area_mode.disδ + 2 * im * τ′ * d.area.dis
            @test result.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result, [d0, d, d0], buffer)
            @test result.τ ≈ τ + τ′ * 2
            @test result.area.dis == d.area.dis
            @test result.area.area == d.area.area
            @test result.cumdis.cumdis ≈ d.cumdis.cumdis + τ′ * d.area.dis
            @test result.area_mode.disδ ≈ d.area_mode.disδ + im * τ′ * d.area.dis
            @test result.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result, [d, d0, d0], buffer)
            @test result.τ ≈ τ + τ′ * 2
            @test result.area.dis == d.area.dis
            @test result.area.area == d.area.area
            @test result.cumdis.cumdis ≈ d.cumdis.cumdis + 2 * τ′ * d.area.dis
            @test result.area_mode.disδ == d.area_mode.disδ
            @test result.area_mode.areaδ == d.area_mode.areaδ
        end
    end
end

@testset "Average zero" begin
    T = Float64
    CT = Complex{T}
    A = SS.AreaData{T}
    CD = SS.CumDisData{T,CT}
    AG = SS.AreaModeData{T,CT}

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SeqResultData{T,A,CD,AG}()

    p0 = 0
    p1 = 1 + 1im
    p2 = 0 - 2im
    p3 = -1 + 1im
    p4 = 0

    l1 = p1 - p0
    l2 = p2 - p1
    l3 = p3 - p2
    l4 = p4 - p3

    for (τ, Ω, φ) in Iterators.product(τs, Ωs, φs)
        d1 = SL.SegInt.compute_values(τ, Ω * abs(l1), 0, φ + angle(l1), 0,
                                      Val(true), Val(true), Val(false))
        d2 = SL.SegInt.compute_values(τ, Ω * abs(l2), 0, φ + angle(l2), 0,
                                      Val(true), Val(true), Val(false))
        d3 = SL.SegInt.compute_values(τ, Ω * abs(l3), 0, φ + angle(l3), 0,
                                      Val(true), Val(true), Val(false))
        d4 = SL.SegInt.compute_values(τ, Ω * abs(l4), 0, φ + angle(l4), 0,
                                      Val(true), Val(true), Val(false))

        SS.compute_sequence!(result, [d1, d2, d3, d4], buffer)
        @test result.τ == 4 * τ
        @test result.area.dis ≈ 0 atol=1e-10
        @test result.area.area ≈ -(Ω * τ)^2 * 4
        @test result.cumdis.cumdis ≈ 0 atol=1e-10
        @test result.area_mode.disδ ≈ 0 atol=1e-10
        if τ != 0 && Ω != 0
            @test result.area_mode.areaδ != 0
        end
    end
end

end
