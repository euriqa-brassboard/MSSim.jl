#!/usr/bin/julia

using Test
using QuadGK
using ForwardDiff

using MSSim: FarDetune as FD

function numeric_area(τ, Ω11, Ω12, Ω21, Ω22, δ; atol=1e-8, rtol=1e-8)
    Ω1(t) = Ω11 + (Ω12 - Ω11) * (t / τ)
    Ω2(t) = Ω21 + (Ω22 - Ω21) * (t / τ)
    f(t) = Ω1(t) * Ω2(t) / δ / 2
    res, err = quadgk(f, 0, τ; atol=atol, rtol=rtol)
    return res
end

@testset "FarDetune" begin
    ωs0 = zeros(1)
    ωs0_2 = zeros(2)
    weights0 = ones(1)
    weights0_2 = ones(2)

    grad = zeros(6)
    grad2 = zeros(6)
    for _ in 1:200
        τ = rand()
        Ω11 = rand() - 0.5
        Ω12 = rand() - 0.5
        Ω21 = rand() - 0.5
        Ω22 = rand() - 0.5
        δ = rand() + 0.5

        vn = numeric_area(τ, Ω11, Ω12, Ω21, Ω22, δ)
        vs = FD.enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ, ())
        @test vs ≈ vn

        vs_autodiff = FD.enclosed_area_kernel(
            ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω11, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω12, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω21, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω22, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
            ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)), ())
        @test vs_autodiff.value ≈ vs

        vs_grad = FD.enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ, grad)
        @test vs_grad ≈ vs
        for i in 1:6
            @test grad[i] ≈ vs_autodiff.partials[i]
        end

        vs2_grad = FD.enclosed_area_modes(τ, Ω11, Ω12, Ω21, Ω22, δ, ωs0, weights0, grad2)
        @test vs2_grad ≈ vs
        @test grad2 ≈ grad

        vs2_grad = FD.enclosed_area_modes(τ, Ω11, Ω12, Ω21, Ω22, δ,
                                          ωs0_2, weights0_2, grad2)
        @test vs2_grad ≈ 2 * vs
        @test grad2 ≈ 2 .* grad

        ωs = rand(5) .- 0.8
        weights = rand(5) .- 0.5
        vs2 = 0.0
        for (ω, weight) in zip(ωs, weights)
            vs2 += FD.enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ - ω, ()) * weight
        end
        vs3 = FD.enclosed_area_modes(τ, Ω11, Ω12, Ω21, Ω22, δ, ωs, weights, ())
        @test vs3 ≈ vs2

        vs2_autodiff = FD.enclosed_area_modes(
            ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω11, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω12, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω21, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω22, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
            ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)), ωs, weights, ())
        @test vs2_autodiff.value ≈ vs2

        vs2_grad = FD.enclosed_area_modes(τ, Ω11, Ω12, Ω21, Ω22, δ, ωs, weights, grad2)
        @test vs2_grad ≈ vs2
        for i in 1:6
            @test grad2[i] ≈ vs2_autodiff.partials[i]
        end
    end
end
