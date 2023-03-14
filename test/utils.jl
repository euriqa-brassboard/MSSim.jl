#!/usr/bin/julia

module Utils

using Test
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
    x = range(0, 1, 100001)[2:end]
    v = f.(x)
    vbig = f_big.(big.(x))

    diff = abs.(Float64.(v .- vbig))
    @show maximum(diff)
    @test all(diff .<= threshold)
end

@testset "sinc" begin
    test_diffs(MSSim.Utils.sinc, MSSim.Utils._sinc_big)
end

@testset "cosc" begin
    test_diffs(MSSim.Utils.cosc, MSSim.Utils._cosc_big)
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
