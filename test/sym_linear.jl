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
        v_sym = SL.SegInt.displacement(ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0)),
                                       ForwardDiff.Dual(Ω, (0.0, 1.0, 0.0, 0.0, 0.0)),
                                       ForwardDiff.Dual(Ω′, (0.0, 0.0, 1.0, 0.0, 0.0)),
                                       ForwardDiff.Dual(φ, (0.0, 0.0, 0.0, 1.0, 0.0)),
                                       ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 1.0)))
        v_sym_v = get_value(v_sym)
        v_sym_τ = get_partial(v_sym, 1)
        v_sym_Ω = get_partial(v_sym, 2)
        v_sym_Ω′ = get_partial(v_sym, 3)
        v_sym_φ = get_partial(v_sym, 4)
        v_sym_δ = get_partial(v_sym, 5)
        vgrad_sym = SL.SegInt.displacement_gradients(τ, Ω, Ω′, φ, δ)

        vδ_sym = SL.SegInt.displacement_δ(
            ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω, (0.0, 1.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω′, (0.0, 0.0, 1.0, 0.0, 0.0)),
            ForwardDiff.Dual(φ, (0.0, 0.0, 0.0, 1.0, 0.0)),
            ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 1.0)))
        vδ_sym_v = get_value(vδ_sym)
        vδ_sym_τ = get_partial(vδ_sym, 1)
        vδ_sym_Ω = get_partial(vδ_sym, 2)
        vδ_sym_Ω′ = get_partial(vδ_sym, 3)
        vδ_sym_φ = get_partial(vδ_sym, 4)
        vδ_sym_δ = get_partial(vδ_sym, 5)
        vδgrad_sym = SL.SegInt.displacement_δ_gradients(τ, Ω, Ω′, φ, δ)

        @test v_num ≈ v_sym_v atol = 1e-12 rtol = 1e-12
        @test v_sym_τ ≈ vgrad_sym[1] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω ≈ vgrad_sym[2] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω′ ≈ vgrad_sym[3] atol = 1e-12 rtol = 1e-12
        @test v_sym_φ ≈ vgrad_sym[4] atol = 1e-12 rtol = 1e-12
        @test v_sym_δ ≈ vgrad_sym[5] atol = 1e-12 rtol = 1e-12

        @test v_sym_δ ≈ vδ_sym_v atol = 1e-12 rtol = 1e-12
        @test vδ_sym_τ ≈ vδgrad_sym[1] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_Ω ≈ vδgrad_sym[2] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_Ω′ ≈ vδgrad_sym[3] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_φ ≈ vδgrad_sym[4] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_δ ≈ vδgrad_sym[5] atol = 1e-12 rtol = 1e-12
    end
end

@testset "Cumulative Displacement" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        Ωf, θf = get_Ω_θ_func(Ω, Ω′, φ, δ)
        v_num = PN.cumulative_displacement(0, τ, Ωf, θf)
        v_sym = SL.SegInt.cumulative_displacement(
            ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω, (0.0, 1.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω′, (0.0, 0.0, 1.0, 0.0, 0.0)),
            ForwardDiff.Dual(φ, (0.0, 0.0, 0.0, 1.0, 0.0)),
            ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 1.0)))
        v_sym_v = get_value(v_sym)
        v_sym_τ = get_partial(v_sym, 1)
        v_sym_Ω = get_partial(v_sym, 2)
        v_sym_Ω′ = get_partial(v_sym, 3)
        v_sym_φ = get_partial(v_sym, 4)
        v_sym_δ = get_partial(v_sym, 5)
        vgrad_sym = SL.SegInt.cumulative_displacement_gradients(τ, Ω, Ω′, φ, δ)

        @test v_num ≈ v_sym_v atol = 1e-12 rtol = 1e-12
        @test v_sym_τ ≈ vgrad_sym[1] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω ≈ vgrad_sym[2] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω′ ≈ vgrad_sym[3] atol = 1e-12 rtol = 1e-12
        @test v_sym_φ ≈ vgrad_sym[4] atol = 1e-12 rtol = 1e-12
        @test v_sym_δ ≈ vgrad_sym[5] atol = 1e-11 rtol = 1e-11
    end
end

@testset "Enclosed Area" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        Ωf, θf = get_Ω_θ_func(Ω, Ω′, φ, δ)
        v_num = PN.enclosed_area(0, τ, Ωf, θf)
        v_sym = SL.SegInt.enclosed_area(ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0)),
                                        ForwardDiff.Dual(Ω, (0.0, 1.0, 0.0, 0.0, 0.0)),
                                        ForwardDiff.Dual(Ω′, (0.0, 0.0, 1.0, 0.0, 0.0)),
                                        ForwardDiff.Dual(φ, (0.0, 0.0, 0.0, 1.0, 0.0)),
                                        ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 1.0)))
        v_sym_v = get_value(v_sym)
        v_sym_τ = get_partial(v_sym, 1)
        v_sym_Ω = get_partial(v_sym, 2)
        v_sym_Ω′ = get_partial(v_sym, 3)
        v_sym_φ = get_partial(v_sym, 4)
        v_sym_δ = get_partial(v_sym, 5)
        vgrad_sym = SL.SegInt.enclosed_area_gradients(τ, Ω, Ω′, φ, δ)

        vδ_sym = SL.SegInt.enclosed_area_δ(
            ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω, (0.0, 1.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω′, (0.0, 0.0, 1.0, 0.0, 0.0)),
            ForwardDiff.Dual(φ, (0.0, 0.0, 0.0, 1.0, 0.0)),
            ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 1.0)))
        vδ_sym_v = get_value(vδ_sym)
        vδ_sym_τ = get_partial(vδ_sym, 1)
        vδ_sym_Ω = get_partial(vδ_sym, 2)
        vδ_sym_Ω′ = get_partial(vδ_sym, 3)
        vδ_sym_φ = get_partial(vδ_sym, 4)
        vδ_sym_δ = get_partial(vδ_sym, 5)
        vδgrad_sym = SL.SegInt.enclosed_area_δ_gradients(τ, Ω, Ω′, φ, δ)

        @test v_num ≈ v_sym_v atol = 2e-11 rtol = 1e-11
        @test v_sym_τ ≈ vgrad_sym[1] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω ≈ vgrad_sym[2] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω′ ≈ vgrad_sym[3] atol = 1e-12 rtol = 1e-12
        @test v_sym_φ == vgrad_sym[4] == 0
        @test v_sym_δ ≈ vgrad_sym[5] atol = 1e-12 rtol = 1e-12

        @test v_sym_δ ≈ vδ_sym_v atol = 1e-12 rtol = 1e-12
        @test vδ_sym_τ ≈ vδgrad_sym[1] atol = 2e-11 rtol = 1e-11
        @test vδ_sym_Ω ≈ vδgrad_sym[2] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_Ω′ ≈ vδgrad_sym[3] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_φ == vδgrad_sym[4] == 0
        @test vδ_sym_δ ≈ vδgrad_sym[5] atol = 1e-12 rtol = 1e-12

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
        v_dis_grad = SL.SegInt.displacement_gradients(τ, Ω, Ω′, φ, δ)
        v_disδ_grad = SL.SegInt.displacement_δ_gradients(τ, Ω, Ω′, φ, δ)
        v_cumdis = SL.SegInt.cumulative_displacement(τ, Ω, Ω′, φ, δ)
        v_cumdis_grad = SL.SegInt.cumulative_displacement_gradients(τ, Ω, Ω′, φ, δ)
        v_area = SL.SegInt.enclosed_area(τ, Ω, Ω′, φ, δ)
        v_areaδ = SL.SegInt.enclosed_area_δ(τ, Ω, Ω′, φ, δ)
        v_area_grad = SL.SegInt.enclosed_area_gradients(τ, Ω, Ω′, φ, δ)
        v_areaδ_grad = SL.SegInt.enclosed_area_δ_gradients(τ, Ω, Ω′, φ, δ)

        for (include_cumdis, include_area_mode, include_grad) in
            Iterators.product((false, true), (false, true), (false, true))

            d, area_grad, cumdis_grad, area_mode_grad =
                SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(include_cumdis),
                                         Val(include_area_mode), Val(include_grad))
            @test d.area.dis ≈ v_dis
            @test d.area.area ≈ v_area
            if include_cumdis
                @test d.cumdis.cumdis ≈ v_cumdis
            end
            if include_area_mode
                @test d.area_mode.disδ ≈ v_disδ
                @test d.area_mode.areaδ ≈ v_areaδ
            end
            if include_grad
                @test length(area_grad) == 5
                @test area_grad[1].dis ≈ v_dis_grad[1]
                @test area_grad[2].dis ≈ v_dis_grad[2]
                @test area_grad[3].dis ≈ v_dis_grad[3]
                @test area_grad[4].dis ≈ v_dis_grad[4]
                @test area_grad[5].dis ≈ v_dis_grad[5]
                @test area_grad[1].area ≈ v_area_grad[1]
                @test area_grad[2].area ≈ v_area_grad[2]
                @test area_grad[3].area ≈ v_area_grad[3]
                @test area_grad[4].area ≈ v_area_grad[4]
                @test area_grad[5].area ≈ v_area_grad[5]

                @test length(cumdis_grad) == 5
                if include_cumdis
                    @test cumdis_grad[1].cumdis ≈ v_cumdis_grad[1]
                    @test cumdis_grad[2].cumdis ≈ v_cumdis_grad[2]
                    @test cumdis_grad[3].cumdis ≈ v_cumdis_grad[3]
                    @test cumdis_grad[4].cumdis ≈ v_cumdis_grad[4]
                    @test cumdis_grad[5].cumdis ≈ v_cumdis_grad[5]
                end

                @test length(area_mode_grad) == 5
                if include_area_mode
                    @test area_mode_grad[1].disδ ≈ v_disδ_grad[1]
                    @test area_mode_grad[2].disδ ≈ v_disδ_grad[2]
                    @test area_mode_grad[3].disδ ≈ v_disδ_grad[3]
                    @test area_mode_grad[4].disδ ≈ v_disδ_grad[4]
                    @test area_mode_grad[5].disδ ≈ v_disδ_grad[5]
                    @test area_mode_grad[1].areaδ ≈ v_areaδ_grad[1]
                    @test area_mode_grad[2].areaδ ≈ v_areaδ_grad[2]
                    @test area_mode_grad[3].areaδ ≈ v_areaδ_grad[3]
                    @test area_mode_grad[4].areaδ ≈ v_areaδ_grad[4]
                    @test area_mode_grad[5].areaδ ≈ v_areaδ_grad[5]
                end
            else
                @test area_grad === nothing
                @test cumdis_grad === nothing
                @test area_mode_grad === nothing
            end
        end
    end
end

end
