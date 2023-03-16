#!/usr/bin/julia

module Utils

using Test
using ForwardDiff
using MSSim

function test_diffs(_f, _f_big, threshold=1e-15)
    function f_big(x)
        s, c = sincos(x)
        return _f_big(x, s, c)
    end
    function f(x)
        s, c = sincos(x)
        return _f(x, s, c)
    end
    xs = range(0, 1, 100001)[2:end]
    v = f.(xs)
    vbig = f_big.(big.(xs))
    err = abs.(Float64.(v .- vbig))
    @test all(err .<= threshold)

    diff = [ForwardDiff.derivative(f, x) for x in xs]
    diff_big = [ForwardDiff.derivative(f_big, big(x)) for x in xs]
    diff_err = abs.(Float64.(diff .- diff_big))
    @test all(diff_err .<= threshold * 8)
end

@testset "sin_c1" begin
    test_diffs(MSSim.Utils.sin_c1, MSSim.Utils._sin_c1_big)
end

@testset "sin_c2" begin
    test_diffs(MSSim.Utils.sin_c2, MSSim.Utils._sin_c2_big)
end

@testset "cos_f1" begin
    test_diffs(MSSim.Utils.cos_f1, MSSim.Utils._cos_f1_big)
end

@testset "sin_f1" begin
    test_diffs(MSSim.Utils.sin_f1, MSSim.Utils._sin_f1_big)
end

@testset "cos_f2" begin
    test_diffs(MSSim.Utils.cos_f2, MSSim.Utils._cos_f2_big)
end

@testset "sin_f2" begin
    test_diffs(MSSim.Utils.sin_f2, MSSim.Utils._sin_f2_big)
end

@testset "cos_f3" begin
    test_diffs(MSSim.Utils.cos_f3, MSSim.Utils._cos_f3_big)
end

@testset "sin_f3" begin
    test_diffs(MSSim.Utils.sin_f3, MSSim.Utils._sin_f3_big)
end

end
