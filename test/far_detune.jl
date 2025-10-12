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
    grad = zeros(6)
    for _ in 1:1000
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
    end
end
