#!/usr/bin/julia

using Test

using LinearAlgebra

import MSSim.Optimizers as Opts
import MSSim.SymLinear as SL
import MSSim.Sequence as Seq

function compute_grad(v₋₄, v₋₃, v₋₂, v₋₁, v₁, v₂, v₃, v₄, h)
    return (-(v₄ - v₋₄) / 280 + 4 * (v₃ - v₋₃) / 105
            - (v₂ - v₋₂) / 5 + 4 * (v₁ - v₋₁) / 5) / h
end

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

@testset "Abs Area Objective" begin
    modes = Seq.Modes()
    for i in 1:5
        push!(modes, (2.1 + 0.1 * i) * 2π)
    end
    obj = Opts.abs_area_obj(10, modes, SL.pmask_tfm)
    @test obj.obj.dis_weights == [1, 1, 1, 1, 1]
    @test obj.obj.disδ_weights == [1, 1, 1, 1, 1]
    @test obj.obj.area_weights == [1, 1, 1, 1, 1]

    obj = Opts.abs_area_obj(10, modes, SL.pmask_tfm,
                            dis_weights=2, disδ_weights=0.1, area_weights=0.3)
    @test obj.obj.dis_weights == [2, 2, 2, 2, 2]
    @test obj.obj.disδ_weights == [0.1, 0.1, 0.1, 0.1, 0.1]
    @test obj.obj.area_weights == [0.3, 0.3, 0.3, 0.3, 0.3]

    dis_weights = rand(5)
    disδ_weights = rand(5)
    area_weights = rand(5)
    obj = Opts.abs_area_obj(10, modes, SL.pmask_tfm,
                            dis_weights=dis_weights, disδ_weights=disδ_weights,
                            area_weights=area_weights)
    @test obj.obj.dis_weights == dis_weights
    @test obj.obj.disδ_weights == disδ_weights
    @test obj.obj.area_weights == area_weights

    args = Vector{Float64}(undef, 15)
    args2 = Vector{Float64}(undef, 15)
    grads = Vector{Float64}(undef, 15)
    grads2 = Vector{Float64}(undef, 15)
    for _ in 1:10
        dis_weights = rand(5)
        disδ_weights = rand(5)
        area_weights = rand(5)
        obj.obj.dis_weights .= dis_weights
        obj.obj.disδ_weights .= disδ_weights
        obj.obj.area_weights .= area_weights

        for _ in 1:100
            dis = rand(5)
            disδ = rand(5)
            area = (rand(5) .+ 0.1) .* rand((1, -1), 5)

            args[1:5] .= dis
            args[6:10] .= disδ
            args[11:15] .= area
            args2 .= args

            function eval_wrapper(i, d)
                args2[i] = args[i] + d
                return obj.obj(args2, grads2)
            end

            @test obj.obj(args, grads) ≈
                (dot(dis, dis_weights) + dot(disδ, disδ_weights)) / dot(abs.(area), area_weights)

            h = 0.000005 / 4
            hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
            for i in 1:15
                results = eval_wrapper.(i, hs)
                args2[i] = args[i]
                @test grads[i] ≈ compute_grad(results..., h) rtol=1.5e-5 atol=5e-5
            end
        end
    end
end
