#!/usr/bin/julia

using Test

using MSSim
const Opts = MSSim.Optimizers

@testset "NLVarTracker" begin
    tracker = Opts.NLVarTracker(10)
    @test isequal(Opts.lower_bounds(tracker), fill(-Inf, 10))
    @test isequal(Opts.upper_bounds(tracker), fill(Inf, 10))
    for _ in 1:100
        @test all(0 .<= Opts.init_vars!(tracker) .<= 1)
    end
    for i in 1:10
        Opts.set_bound!(tracker, i, -0.1 * i, 2 * i)
    end
    lb = [-0.1 .* (1:10);]
    ub = [2.0 .* (1:10);]
    @test Opts.lower_bounds(tracker) ≈ lb
    @test Opts.upper_bounds(tracker) == ub
    for _ in 1:100
        vars = zeros(10)
        @test Opts.init_vars!(tracker, vars) === vars
        @test all(lb .<= vars .<= ub)
    end
    Opts.set_bound!(tracker, 2, -Inf, 2.3)
    lb[2] = 1.3
    ub[2] = 2.3
    Opts.set_bound!(tracker, 5, 1.2, Inf)
    lb[5] = 1.2
    ub[5] = 2.2
    for _ in 1:100
        vars = zeros(10)
        @test Opts.init_vars!(tracker, vars) === vars
        @test all(lb .<= vars .<= ub)
    end
end

@testset "autodiff" begin
    func(x) = x[1] * x[2] + x[3] * 2
    func_diff = Opts.autodiff(func)
    for _ in 1:100
        xs = rand(3)
        gs = zeros(3)
        @test func_diff(xs, gs) == func(xs)
        @test gs ≈ [xs[2], xs[1], 2]
    end
end
