#!/usr/bin/julia

using Test

using LinearAlgebra
using Random

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

@testset "Target Area Objective" begin
    modes = Seq.Modes()
    for i in 1:5
        push!(modes, (2.1 + 0.1 * i) * 2π)
    end
    obj = Opts.target_area_obj(10, modes, SL.pmask_tfm, area_targets=Opts.AreaTarget[])
    @test length(obj.obj.params) == 10
    @test obj.obj.dis_weights == [1, 1, 1, 1, 1]
    @test obj.obj.disδ_weights == [1, 1, 1, 1, 1]
    @test obj.obj.area_targets == ()

    obj = Opts.target_area_obj(10, modes, SL.pmask_tfm, area_targets=Opts.AreaTarget[],
                               dis_weights=2, disδ_weights=0.1)
    @test obj.obj.dis_weights == [2, 2, 2, 2, 2]
    @test obj.obj.disδ_weights == [0.1, 0.1, 0.1, 0.1, 0.1]
    @test obj.obj.area_targets == ()

    dis_weights = rand(5)
    disδ_weights = rand(5)
    obj = Opts.target_area_obj(10, modes, SL.pmask_tfm,
                               area_targets=[Opts.AreaTarget(1, target=0.1,
                                                             area_weights=[0.1, -0.2])],
                               dis_weights=dis_weights, disδ_weights=disδ_weights)
    @test length(obj.obj.params) == 13
    @test obj.obj.dis_weights == dis_weights
    @test obj.obj.disδ_weights == disδ_weights
    @test length(obj.obj.area_targets) == 1
    area_tgt = obj.obj.area_targets[1]
    @test area_tgt.target == 0.1
    @test area_tgt.area_weights == [0.1, -0.2]
    @test area_tgt.areaδ_weights === nothing
    @test area_tgt._obj === obj.obj

    area_tgt.target = 2.2
    @test area_tgt.target == 2.2
    area_tgt.area_weights .= [1.0, 3.0]
    @test area_tgt.area_weights == [1.0, 3.0]

    obj = Opts.target_area_obj(10, modes, SL.pmask_tfm,
                               area_targets=[Opts.AreaTarget(1, target=0.1,
                                                             area_weights=[0.2, -0.2]),
                                             Opts.AreaTarget(2, target=0.9,
                                                             area_weights=[0.5, -0.3],
                                                             areaδ_weights=[0.8, -0.7])],
                               dis_weights=dis_weights, disδ_weights=disδ_weights)
    @test length(obj.obj.params) == 18
    @test obj.obj.dis_weights == dis_weights
    @test obj.obj.disδ_weights == disδ_weights
    @test length(obj.obj.area_targets) == 2
    area_tgt1 = obj.obj.area_targets[1]
    @test area_tgt1.target == 0.1
    @test area_tgt1.area_weights == [0.2, -0.2]
    @test area_tgt1.areaδ_weights === nothing
    @test area_tgt1._obj === obj.obj
    area_tgt2 = obj.obj.area_targets[2]
    @test area_tgt2.target == 0.9
    @test area_tgt2.area_weights == [0.5, -0.3]
    @test area_tgt2.areaδ_weights == [0.8, -0.7]
    @test area_tgt2._obj === obj.obj

    area_tgt1.target = 2.2
    @test area_tgt1.target == 2.2
    area_tgt1.area_weights .= [1.0, 3.0]
    @test area_tgt1.area_weights == [1.0, 3.0]

    area_tgt2.target = 2.9
    @test area_tgt2.target == 2.9
    area_tgt2.area_weights .= [-1.0, 3.9]
    @test area_tgt2.area_weights == [-1.0, 3.9]
    area_tgt2.areaδ_weights .= [2.9, 3.1]
    @test area_tgt2.areaδ_weights == [2.9, 3.1]

    function test_area_target(target_info)
        area_targets = Opts.AreaTarget[]
        nmodes = length(modes.modes)
        nparams = nmodes * 2
        dis_weights = rand(nmodes) .* 2 .- 1
        disδ_weights = rand(nmodes) .* 2 .- 1
        for (a_start, a_modes, hasδ) in target_info
            push!(area_targets, Opts.AreaTarget(a_start, target=rand(),
                                                area_weights=rand(a_modes) .* 2 .- 1,
                                                areaδ_weights=hasδ ?
                                                    rand(a_modes) .* 2 .- 1 :
                                                    Float64[]))
            nparams += 1 + (hasδ + 1) * a_modes
        end
        function obj_func(args)
            dis = @view args[1:nmodes]
            disδ = @view args[nmodes + 1:2 * nmodes]
            area = @view args[2 * nmodes + 1:3 * nmodes]
            areaδ = @view args[3 * nmodes + 1:4 * nmodes]
            res = dot(dis, dis_weights) + dot(disδ, disδ_weights)
            for area_tgt in area_targets
                a_modes = length(area_tgt.area_weights)
                start_idx = area_tgt.start_idx
                res += (abs(dot(@view(area[start_idx:start_idx + a_modes - 1]),
                                area_tgt.area_weights)) - area_tgt.target)^2
                if !isempty(area_tgt.areaδ_weights)
                    res += dot(@view(areaδ[start_idx:start_idx + a_modes - 1]),
                               area_tgt.areaδ_weights)^2
                end
            end
            return res
        end
        obj = Opts.target_area_obj(10, modes, SL.pmask_full,
                                   dis_weights=dis_weights, disδ_weights=disδ_weights,
                                   area_targets=area_targets).obj
        @test length(obj.params) == nparams
        args = Vector{Float64}(undef, nmodes * 4)
        args2 = Vector{Float64}(undef, nmodes * 4)
        grads = Vector{Float64}(undef, nmodes * 4)
        grads2 = Vector{Float64}(undef, nmodes * 4)
        function eval_wrapper(i, d)
            args2[i] = args[i] + d
            return obj(args2, grads2)
        end
        for _ in 1:100
            rand!(args)
            args .= args .* 2 .- 1
            @test obj(args, grads) ≈ obj_func(args)

            args2 .= args
            h = 0.000005 / 4
            hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
            for i in 1:(nmodes * 4)
                results = eval_wrapper.(i, hs)
                args2[i] = args[i]
                @test grads[i] ≈ compute_grad(results..., h) rtol=1.5e-5 atol=5e-5
            end
        end
    end
    test_area_target([])
    test_area_target([(1, 2, true), (3, 2, false)])
    test_area_target([(1, 5, false), (2, 3, true)])
    test_area_target([(1, 4, true), (2, 4, true), (1, 2, false)])
end
