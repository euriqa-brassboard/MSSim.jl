#!/usr/bin/julia

module Utils

using Test
using ForwardDiff
using MSSim
const U = MSSim.Utils

function test_diffs(_f, _f_big, threshold=1e-15)
    function f_big(x)
        s, c = sincos(x)
        return _f_big(x, s, c)
    end
    function f(x)
        s, c = sincos(x)
        return _f(x, s, c)
    end
    xs = range(0, 3, 300001)[2:end]
    for x in xs
        @test f(x) ≈ f_big(big(x)) atol=threshold rtol=0
        @test(ForwardDiff.derivative(f, x) ≈ ForwardDiff.derivative(f_big, big(x)),
              atol=threshold * 8, rtol=0)
    end
end

@testset "mulim" begin
    for x in (0, 2, 1.2, 2im, 1.5im, 3 - 2im, 5.6 + 3.8im)
        @test U.mulim(x) == im * x
    end
end

@testset "sin_c1" begin
    test_diffs(U.sin_c1, U._sin_c1_big)
end

@testset "sin_c2" begin
    test_diffs(U.sin_c2, U._sin_c2_big)
end

@testset "sin_c3" begin
    test_diffs(U.sin_c3, U._sin_c3_big)
end

@testset "cos_f1" begin
    test_diffs(U.cos_f1, U._cos_f1_big)
end

@testset "sin_f1" begin
    test_diffs(U.sin_f1, U._sin_f1_big)
end

@testset "cos_f2" begin
    test_diffs(U.cos_f2, U._cos_f2_big)
end

@testset "sin_f2" begin
    test_diffs(U.sin_f2, U._sin_f2_big)
end

@testset "cos_f3" begin
    test_diffs(U.cos_f3, U._cos_f3_big)
end

@testset "sin_f3" begin
    test_diffs(U.sin_f3, U._sin_f3_big)
end

@testset "sin_f4" begin
    test_diffs(U.sin_f4, U._sin_f4_big)
end

@testset "sin_f5" begin
    test_diffs(U.sin_f5, U._sin_f5_big)
end

end
