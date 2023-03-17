#!/usr/bin/julia

module SymLinear

using MSSim
const PN = MSSim.PureNumeric
const SL = MSSim.SymLinear
const SegSeq = MSSim.SegSeq

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

@testset "Compute values" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        v_dis = SL.SegInt.displacement(τ, Ω, Ω′, φ, δ)
        v_disδ = SL.SegInt.displacement_δ(τ, Ω, Ω′, φ, δ)
        v_cumdis = SL.SegInt.cumulative_displacement(τ, Ω, Ω′, φ, δ)
        v_area = SL.SegInt.enclosed_area(τ, Ω, Ω′, φ, δ)
        v_areaδ = SL.SegInt.enclosed_area_δ(τ, Ω, Ω′, φ, δ)

        for ((T_CD, CT_CD),
             (T_AG, CT_AG)) in Iterators.product(((Nothing, Nothing),
                                                  (Float64, ComplexF64)),
                                                 ((Nothing, Nothing),
                                                  (Float64, ComplexF64)))
            CD = SegSeq.CumDisData{T_CD,CT_CD}
            AG = SegSeq.AreaModeData{T_AG,CT_AG}
            area, cumdis, area_mode, area_grad, cumdis_grad, area_mode_grad =
                SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ,
                                         SegSeq.AreaData{Float64}, CD, AG, Val(false))
            @test area.dis ≈ v_dis
            @test area.area ≈ v_area
            if T_CD !== Nothing
                @test cumdis.cumdis ≈ v_cumdis
            end
            if T_AG !== Nothing
                @test area_mode.disδ ≈ v_disδ
                @test area_mode.areaδ ≈ v_areaδ
            end
            @test area_grad == ()
            @test cumdis_grad == ()
            @test area_mode_grad == ()
        end
    end
end

end
