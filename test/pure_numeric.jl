#!/usr/bin/julia

using MSSim: PureNumeric as PN

using Test
using QuadGK

@testset "Displacement" begin
    @test PN.displacement(0, 3.1, t->1.2, t->0.1) ≈ 3.1 * 1.2 * exp(im * 0.1)
    @test PN.displacement(0, 3.1, t->1.2 - 0.5 * t, t->0.1) ≈
        3.1 * (1.2 - 0.5 * 3.1 / 2) * exp(im * 0.1)
    @test PN.displacement(0, 3.1, t->1.2, t->0.1 + (2π / 3.1) * t) ≈ 0 atol=1e-13
end

@testset "Cumulative Displacement" begin
    @test PN.cumulative_displacement(0, 3.1, t->1.2, t->0.1) ≈
        3.1 * 1.2 * exp(im * 0.1) * 3.1 / 2
    @test PN.cumulative_displacement(0, 3.1, t->1.2 - 0.5 * t, t->0.1) ≈
        3.1 * (1.2 / 2 - 0.5 * 3.1 / 6) * exp(im * 0.1) * 3.1
    @test PN.cumulative_displacement(0, 3.1, t->1.2, t->0.1 + (2π / 3.1) * t) ≈
        im * 1.2 * 3.1^2 / (2π) * exp(im * 0.1)
end

@testset "Enclosed Area" begin
    @test PN.enclosed_area(0, 3.1, t->1.2, t->0.1 + (2π / 3.1) * t) ≈
        1.2^2 * 3.1^2 / (2π)
    @test PN.enclosed_area(0, 1.5 * 2, t->0.2, t->0.5 + (2π / 1.5) * t) ≈
        0.2^2 * 1.5^2 / (2π) * 2
    @test PN.enclosed_area_complex(0, 3.1, t->1.2, t->0.1 + (2π / 3.1) * t) ≈
        im * 1.2^2 * 3.1^2 / (2π)
end
