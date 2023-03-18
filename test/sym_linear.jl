#!/usr/bin/julia

module SymLinear

using MSSim
const PN = MSSim.PureNumeric
const SL = MSSim.SymLinear

using Test
using ForwardDiff

function get_Ω_θ_func(Ω, Ω′, φ, δ)
    return t->Ω + Ω′ * t, t->φ + δ * t
end

get_value(v) = v.value
get_value(v::Complex) = complex(get_value(real(v)), get_value(imag(v)))

get_partial(v, i) = v.partials[i]
get_partial(v::Complex, i) = complex(get_partial(real(v), i),
                                     get_partial(imag(v), i))

const τs = [0, 1, 5, 10, 20]
const Ωs = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0, 0.001, 0.02, 1, 5, 10, 20]
const Ω′s = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0,
               0.001, 0.02, 1, 5, 10, 20] ./ 10
const φs = range(0, 2π, 17)
const δs = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0,
             0.001, 0.02, 1, 5, 10, 20] ./ 10

const all_params = Iterators.product(τs, Ωs, Ω′s, φs, δs)

@testset "Displacement" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        Ωf, θf = get_Ω_θ_func(Ω, Ω′, φ, δ)
        v_num = PN.displacement(0, τ, Ωf, θf)
        v_sym = SL.SegInt.displacement(τ, Ω, Ω′, φ, ForwardDiff.Dual(δ, (1.0,)))
        vd_sym = SL.SegInt.displacement_δ(τ, Ω, Ω′, φ, δ)
        v_sym_v = get_value(v_sym)
        v_sym_d = get_partial(v_sym, 1)
        @test v_num ≈ v_sym_v atol = 1e-12 rtol = 1e-12
        @test v_sym_d ≈ vd_sym atol = 1e-12 rtol = 1e-12
    end
end

@testset "Cumulative Displacement" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        Ωf, θf = get_Ω_θ_func(Ω, Ω′, φ, δ)
        v_num = PN.cumulative_displacement(0, τ, Ωf, θf)
        v_sym = SL.SegInt.cumulative_displacement(τ, Ω, Ω′, φ, δ)
        @test v_num ≈ v_sym atol = 1e-12 rtol = 1e-12
    end
end

@testset "Enclosed Area" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        Ωf, θf = get_Ω_θ_func(Ω, Ω′, φ, δ)
        v_num = PN.enclosed_area(0, τ, Ωf, θf)
        v_sym = SL.SegInt.enclosed_area(τ, Ω, Ω′, φ, ForwardDiff.Dual(δ, (1.0,)))
        vd_sym = SL.SegInt.enclosed_area_δ(τ, Ω, Ω′, φ, δ)
        v_sym_v = get_value(v_sym)
        v_sym_d = get_partial(v_sym, 1)
        @test v_num ≈ v_sym_v atol = 1e-10 rtol = 1e-10
        @test v_sym_d ≈ vd_sym atol = 1e-12 rtol = 1e-12
        vc_num = PN.enclosed_area_complex(0, τ, Ωf, θf)
        vc_sym = SL.SegInt.enclosed_area_complex(τ, Ω, Ω′, φ, δ)
        @test vc_num ≈ vc_sym atol = 1e-10 rtol = 1e-10
    end
end

@testset "Special cases" begin
    for (τ, Ω, φ) in Iterators.product(τs, Ωs, φs)
        @test SL.SegInt.displacement(τ, Ω, 0, φ, 0) ≈ τ * Ω * cis(φ)
        @test SL.SegInt.displacement_δ(τ, Ω, 0, φ, 0) ≈ im * τ^2 * Ω * cis(φ) / 2
        @test SL.SegInt.cumulative_displacement(τ, Ω, 0, φ, 0) ≈ τ^2 * Ω * cis(φ) / 2
        @test SL.SegInt.enclosed_area(τ, Ω, 0, φ, 0) == 0
        @test SL.SegInt.enclosed_area_δ(τ, Ω, 0, φ, 0) ≈ abs2(τ * Ω) * τ / 6
    end
end

@testset "Compute values" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        v_dis = SL.SegInt.displacement(τ, Ω, Ω′, φ, δ)
        v_disδ = SL.SegInt.displacement_δ(τ, Ω, Ω′, φ, δ)
        v_cumdis = SL.SegInt.cumulative_displacement(τ, Ω, Ω′, φ, δ)
        v_area = SL.SegInt.enclosed_area(τ, Ω, Ω′, φ, δ)
        v_areaδ = SL.SegInt.enclosed_area_δ(τ, Ω, Ω′, φ, δ)

        for (include_cumdis, include_area_mode) in Iterators.product((false, true),
                                                                     (false, true))
            d = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(include_cumdis),
                                         Val(include_area_mode), Val(false))
            @test d.area.dis ≈ v_dis
            @test d.area.area ≈ v_area
            if include_cumdis
                @test d.cumdis.cumdis ≈ v_cumdis
            end
            if include_area_mode
                @test d.area_mode.disδ ≈ v_disδ
                @test d.area_mode.areaδ ≈ v_areaδ
            end
            @test d.area_grad == ()
            @test d.cumdis_grad == ()
            @test d.area_mode_grad == ()
        end
    end
end

end
