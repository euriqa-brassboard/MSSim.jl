#!/usr/bin/julia

using Test
using ForwardDiff
using MSSim
const U = MSSim.Utils

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

    empty!(m2)
    @test length(m2) == 0
    @test size(m2) == (0,)

    resize!(m2, m)
    @test length(m2) == 2
    @test size(m2) == (2,)
    @test size(m2[1]) == (1,)
    @test size(m2[2]) == (5,)

    resize!(m2, [[1, 2, 3], [2], [3, 4, 5, 6], Int[]])
    @test length(m2) == 4
    @test size(m2) == (4,)
    @test size(m2[1]) == (3,)
    @test size(m2[2]) == (1,)
    @test size(m2[3]) == (4,)
    @test size(m2[4]) == (0,)
end

@testset "mulim" begin
    for x in (0, 2, 1.2, 2im, 1.5im, 3 - 2im, 5.6 + 3.8im)
        @test U.mulim(x) == im * x
    end
end

@testset "mul/add/sub" begin
    for x in (0, 2, 1.2, 2im, 1.5im, 3 - 2im, 5.6 + 3.8im, big(1), big(1.23))
        for y in (0, 2, 1.2, 2im, 1.5im, 3 - 2im, 5.6 + 3.8im, big(1), big(1.23))
            if isa(x, Real) || isa(y, Real)
                @test U.mul(x, y) == x * y
            else
                @test U.mul(x, y) ≈ x * y
            end
            @test U.add(x, y) == x + y
            @test U.sub(x, y) == x - y
            @test U.sub(y, x) == y - x
        end
    end
end

@testset "zero" begin
    z = U.Zero()
    b1 = BigInt(1)
    @test U.Zero() === z
    @test +z === z

    @test z + z === z
    @test 1 + z === 1
    @test z + 1 === 1
    @test b1 + z === b1
    @test z + b1 === b1

    @test z - z === z
    @test 1 - z === 1
    @test z - 1 === -1
    @test b1 - z === b1
    @test z - b1 == -b1

    @test z * z === z
    @test z * 1 === z
    @test 1 * z === z
    @test z * b1 === z
    @test b1 * z === z

    @test z / 1 === z
    @test z / b1 === z
    # the following is wrong mathematically but it fits what we need for performance
    @test z / 0 === z
    @test isequal(z / z, NaN)
    @test isequal(0 / z, NaN)
    @test isequal(1 / z, Inf)

    @test z^0 === z
    @test z^2 === z
    @test 0^z === 1

    @test muladd(z, z, z) === z
    @test muladd(1, z, z) === z
    @test muladd(b1, z, z) === z
    @test muladd(z, 1, z) === z
    @test muladd(z, b1, z) === z
    @test muladd(2, 1, z) === 2
    @test muladd(2, b1, z) == 2
    @test muladd(z, z, 1) === 1
    @test muladd(z, z, b1) === b1
    @test muladd(2, z, 1) === 1
    @test muladd(2, z, b1) === b1
    @test muladd(z, 2, 1) === 1
    @test muladd(z, 2, b1) === b1
end

function test_diffs(_f; threshold=1e-15)
    function f(x)
        s, c = sincos(x)
        return _f(x, s, c)
    end
    xs = range(-3, 3, 300001)
    for x in xs
        @test f(x) ≈ f(big(x)) atol=threshold rtol=0
        @test(ForwardDiff.derivative(f, x) ≈ ForwardDiff.derivative(f, big(x)),
              atol=threshold * 8, rtol=0)
    end
end

@testset "sin_c1" begin
    test_diffs(U.sin_c1)
end

@testset "sin_c2" begin
    test_diffs(U.sin_c2)
end

@testset "sin_c3" begin
    test_diffs(U.sin_c3)
end

@testset "cos_f1" begin
    test_diffs(U.cos_f1)
end

@testset "sin_f1" begin
    test_diffs(U.sin_f1)
end

@testset "cos_f2" begin
    test_diffs(U.cos_f2)
end

@testset "sin_f2" begin
    test_diffs(U.sin_f2)
end

@testset "cos_f3" begin
    test_diffs(U.cos_f3)
end

@testset "sin_f3" begin
    test_diffs(U.sin_f3)
end

@testset "sin_f4" begin
    test_diffs(U.sin_f4)
end

@testset "sin_f5" begin
    test_diffs(U.sin_f5, threshold=1.2e-15)
end
