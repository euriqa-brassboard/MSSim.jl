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

@testset "JaggedMatrix" begin
    m = U.JaggedMatrix{Float64}()
    @test length(m) == 0
    @test size(m) == (0,)
    @test eltype(eltype(m)) == Float64
    @test_throws BoundsError push!(m, 1.0)
    push!(m, [1.0])
    @test length(m) == 1
    @test size(m) == (1,)
    @test m[1] == [1.0]
    @test typeof(m[1]) == eltype(m)
    @test_throws BoundsError m[2]
    push!(m, [3])
    @test length(m) == 2
    @test size(m) == (2,)
    @test m[1] == [1.0]
    @test m[2] == [3.0]
    @test_throws BoundsError m[3]
    push!(m, 4.5)
    @test length(m) == 2
    @test size(m) == (2,)
    @test m[1] == [1.0]
    @test m[2] == [3.0, 4.5]
    @test_throws BoundsError m[3]
    push!(m, 10)
    @test length(m) == 2
    @test size(m) == (2,)
    @test m[1] == [1.0]
    @test m[2] == [3.0, 4.5, 10.0]
    @test_throws BoundsError m[3]

    m2 = similar(m)
    @test length(m2) == 2
    @test size(m2) == (2,)
    @test size(m2[1]) == (1,)
    @test size(m2[2]) == (3,)
    @test_throws BoundsError m2[3]
    push!(m2, 10.5)
    @test size(m2[2]) == (4,)
    @test size(m[2]) == (3,)
    push!(m, 20)
    push!(m, 34)
    @test size(m2[2]) == (4,)
    @test size(m[2]) == (5,)
    push!(m2, [1, 2, 3])
    @test length(m2) == 3
    @test size(m2) == (3,)
    @test size(m2[1]) == (1,)
    @test size(m2[2]) == (4,)
    @test m2[3] == [1.0, 2.0, 3.0]
    @test_throws BoundsError m2[4]
    @test length(m) == 2
    @test size(m) == (2,)
    @test m[1] == [1.0]
    @test m[2] == [3.0, 4.5, 10.0, 20.0, 34.0]
    @test_throws BoundsError m[3]
    m2[1][1] = 3.4
    m2[2][1] = 0.1
    m2[2][2] = 0.3
    m2[2][3] = 0.85
    @test m2[1] == [3.4]
    @test m2[2] == [0.1, 0.3, 0.85, 10.5]
    @test m2[3] == [1.0, 2.0, 3.0]

    m2 = similar(m, AbstractVector{Int})
    @test length(m2) == 2
    @test size(m2) == (2,)
    @test size(m2[1]) == (1,)
    @test size(m2[2]) == (5,)
    @test_throws BoundsError m2[3]
    @test eltype(eltype(m2)) == Int

    m2 = similar(m, Vector{Tuple{Int,Int}})
    @test length(m2) == 2
    @test size(m2) == (2,)
    @test size(m2[1]) == (1,)
    @test size(m2[2]) == (5,)
    @test_throws BoundsError m2[3]
    @test eltype(eltype(m2)) == NTuple{2,Int}
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
