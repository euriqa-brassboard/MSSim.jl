#!/usr/bin/julia

module SegSeq

using MSSim
const U = MSSim.Utils
const PN = MSSim.PureNumeric
const SL = MSSim.SymLinear
const SS = MSSim.SegSeq

using Test

const τs = [0, 1, 5, 10, 20]
const Ωs = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0, 0.001, 0.02, 1, 5, 10, 20]
const Ω′s = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0,
               0.001, 0.02, 1, 5, 10, 20] ./ 10
const φs = range(0, 2π, 17)
const δs = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0,
             0.001, 0.02, 1, 5, 10, 20] ./ 10

const all_params = Iterators.product(τs, Ωs, Ω′s, φs, δs)

@testset "Single Segment" begin
    T = Float64
    mask_full = SS.ValueMask(true, true, true, true, true, true)
    mask_none = zero(SS.ValueMask)

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SingleModeResult{T}(Val(mask_full), Val(mask_none))

    for (τ, Ω, Ω′, φ, δ) in all_params
        d, = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(mask_full), Val(mask_none))
        SS.compute_single_mode!(result, [d], buffer)
        @test result.val.τ == τ
        @test result.val.dis == d.dis
        @test result.val.area == d.area
        @test result.val.cumdis == d.cumdis
        @test result.val.disδ == d.disδ
        @test result.val.areaδ == d.areaδ

        for (need_cumdis, need_area_mode) in Iterators.product((false, true),
                                                               (false, true))
            maskv′ = SS.ValueMask(true, true, true, need_cumdis,
                                   need_area_mode, need_area_mode)

            result′ = SS.SingleModeResult{T}(Val(maskv′), Val(mask_none))
            d′, = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ,
                                            Val(maskv′), Val(mask_none))
            SS.compute_single_mode!(result′, [d′], buffer)
            @test result′.val.τ == τ
            @test result′.val.dis == d.dis
            @test result′.val.area == d.area
            if need_cumdis
                @test result′.val.cumdis ≈ d.cumdis
            end
            if need_area_mode
                @test result′.val.disδ == d.disδ
                @test result′.val.areaδ == d.areaδ
            end
        end
    end
end

@testset "Trivial Segment" begin
    T = Float64
    mask_full = SS.ValueMask(true, true, true, true, true, true)
    mask_none = zero(SS.ValueMask)

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SingleModeResult{T}(Val(mask_full), Val(mask_none))

    for (τ, Ω, Ω′, φ, δ) in all_params
        d, = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(mask_full), Val(mask_none))
        nd, = SL.SegInt.compute_values(τ, Ω + Ω′ * τ, -Ω′, φ + δ * τ + π, -δ,
                                       Val(mask_full), Val(mask_none))
        @test nd.τ == d.τ
        @test nd.dis ≈ -d.dis
        @test nd.area ≈ -d.area
        @test nd.cumdis ≈ d.cumdis - d.dis * τ
        @test nd.disδ ≈ d.disδ - im * τ * d.dis
        @test nd.areaδ ≈ d.areaδ
        SS.compute_single_mode!(result, [d, nd], buffer)
        @test result.val.τ ≈ 2 * τ
        @test result.val.dis ≈ 0 atol=1e-8
        @test result.val.area ≈ 0 atol=1e-8
        @test result.val.cumdis ≈ 2 * d.cumdis
        @test result.val.disδ ≈ 2 * d.disδ - 2 * im * τ * d.dis
        @test result.val.areaδ ≈
            (2 * imag(d.disδ * conj(d.dis))
             - 2 * τ * abs2(d.dis) + 2 * d.areaδ)

        SS.compute_single_mode!(result, [d, d], buffer)
        @test result.val.τ ≈ 2 * τ
        @test result.val.dis ≈ 2 * d.dis atol=1e-8
        @test result.val.area ≈ 2 * d.area atol=1e-8
        @test result.val.cumdis ≈ 2 * d.cumdis + τ * d.dis
        @test result.val.disδ ≈ 2 * d.disδ + im * τ * d.dis
        @test result.val.areaδ ≈ τ * abs2(d.dis) + 2 * d.areaδ

        for (need_cumdis, need_area_mode) in Iterators.product((false, true),
                                                               (false, true))
            maskv′ = SS.ValueMask(true, true, true, need_cumdis,
                                   need_area_mode, need_area_mode)

            result′ = SS.SingleModeResult{T}(Val(maskv′), Val(mask_none))
            d′, = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ,
                                            Val(maskv′), Val(mask_none))
            SS.compute_single_mode!(result′, [d′, d′], buffer)
            @test result′.val.τ == 2 * τ
            @test result′.val.dis ≈ 2 * d.dis atol=1e-8
            @test result′.val.area ≈ 2 * d.area atol=1e-8
            if need_cumdis
                @test result′.val.cumdis ≈ 2 * d.cumdis + τ * d.dis
            end
            if need_area_mode
                @test result′.val.disδ ≈ 2 * d.disδ + im * τ * d.dis
                @test result′.val.areaδ ≈ τ * abs2(d.dis) + 2 * d.areaδ
            end
        end

        for τ′ in τs
            d0, = SL.SegInt.compute_values(τ′, 0.0, 0.0, 0.0, 0.0,
                                           Val(mask_full), Val(mask_none))
            SS.compute_single_mode!(result, [d0, d], buffer)
            @test result.val.τ ≈ τ + τ′
            @test result.val.dis == d.dis
            @test result.val.area == d.area
            @test result.val.cumdis == d.cumdis
            @test result.val.disδ ≈ d.disδ + im * τ′ * d.dis
            @test result.val.areaδ == d.areaδ

            SS.compute_single_mode!(result, [d, d0], buffer)
            @test result.val.τ ≈ τ + τ′
            @test result.val.dis == d.dis
            @test result.val.area == d.area
            @test result.val.cumdis ≈ d.cumdis + τ′ * d.dis
            @test result.val.disδ == d.disδ
            @test result.val.areaδ == d.areaδ

            SS.compute_single_mode!(result, [d0, d0, d], buffer)
            @test result.val.τ ≈ τ + τ′ * 2
            @test result.val.dis == d.dis
            @test result.val.area == d.area
            @test result.val.cumdis == d.cumdis
            @test result.val.disδ ≈ d.disδ + 2 * im * τ′ * d.dis
            @test result.val.areaδ == d.areaδ

            SS.compute_single_mode!(result, [d0, d, d0], buffer)
            @test result.val.τ ≈ τ + τ′ * 2
            @test result.val.dis == d.dis
            @test result.val.area == d.area
            @test result.val.cumdis ≈ d.cumdis + τ′ * d.dis
            @test result.val.disδ ≈ d.disδ + im * τ′ * d.dis
            @test result.val.areaδ == d.areaδ

            SS.compute_single_mode!(result, [d, d0, d0], buffer)
            @test result.val.τ ≈ τ + τ′ * 2
            @test result.val.dis == d.dis
            @test result.val.area == d.area
            @test result.val.cumdis ≈ d.cumdis + 2 * τ′ * d.dis
            @test result.val.disδ == d.disδ
            @test result.val.areaδ == d.areaδ
        end
    end
end

@testset "Average zero" begin
    T = Float64
    mask_full = SS.ValueMask(true, true, true, true, true, true)
    mask_none = zero(SS.ValueMask)

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SingleModeResult{T}(Val(mask_full), Val(mask_none))

    p0 = 0
    p1 = 1 + 1im
    p2 = 0 - 2im
    p3 = -1 + 1im
    p4 = 0

    l1 = p1 - p0
    l2 = p2 - p1
    l3 = p3 - p2
    l4 = p4 - p3

    for (τ, Ω, φ) in Iterators.product(τs, Ωs, φs)
        d1, = SL.SegInt.compute_values(τ, Ω * abs(l1), 0, φ + angle(l1), 0,
                                       Val(mask_full), Val(mask_none))
        d2, = SL.SegInt.compute_values(τ, Ω * abs(l2), 0, φ + angle(l2), 0,
                                       Val(mask_full), Val(mask_none))
        d3, = SL.SegInt.compute_values(τ, Ω * abs(l3), 0, φ + angle(l3), 0,
                                       Val(mask_full), Val(mask_none))
        d4, = SL.SegInt.compute_values(τ, Ω * abs(l4), 0, φ + angle(l4), 0,
                                       Val(mask_full), Val(mask_none))

        SS.compute_single_mode!(result, [d1, d2, d3, d4], buffer)
        @test result.val.τ == 4 * τ
        @test result.val.dis ≈ 0 atol=1e-10
        @test result.val.area ≈ -(Ω * τ)^2 * 4
        @test result.val.cumdis ≈ 0 atol=1e-10
        @test result.val.disδ ≈ 0 atol=1e-10
        if τ != 0 && Ω != 0
            @test result.val.areaδ != 0
        end
    end
end

struct SegParam{T}
    τ::T
    Ω::T
    Ω′::T
    φ::T
    δ::T
end

function get_Ω_θ_func(params::AbstractVector{SegParam{T}}) where T
    nparams = length(params)
    total_times = Vector{T}(undef, length(params) + 1)
    total_times[1] = 0
    cur_time = T(0)
    for (i, param) in enumerate(params)
        cur_time += param.τ
        total_times[i + 1] = cur_time
    end
    function find_segment(t)
        idx = searchsortedfirst(total_times, t) - 1
        t0 = get(total_times, idx, zero(T))
        param = get(params, idx, SegParam{T}(zero(T), zero(T), zero(T),
                                             zero(T), zero(T)))
        return t0, param
    end
    function Ωf(t)
        t0, param = find_segment(t)
        return param.Ω + param.Ω′ * (t - t0)
    end
    function θf(t)
        t0, param = find_segment(t)
        return param.φ + param.δ * (t - t0)
    end
    return Ωf, θf
end

function get_seg_data(params::AbstractVector{SegParam{T}}) where T
    mask_full = SS.ValueMask(true, true, true, true, true, true)
    grads = U.JaggedMatrix{SS.SegData(T, mask_full)}()
    function compute_values(param)
        res, grad = SL.SegInt.compute_values(param.τ, param.Ω, param.Ω′, param.φ,
                                             param.δ, Val(mask_full), Val(mask_full))
        push!(grads, grad)
        return res
    end
    return [compute_values(param) for param in params], grads
end

@testset "Random sequence" begin
    T = Float64
    mask_full = SS.ValueMask(true, true, true, true, true, true)
    mask_none = zero(SS.ValueMask)

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SingleModeResult{T}(Val(mask_full), Val(mask_none))

    all_params_array = [SegParam{T}(τ, Ω, Ω′, φ, δ) for (τ, Ω, Ω′, φ, δ) in all_params]

    function test_nseg(nseg)
        params = [rand(all_params_array) for i in 1:nseg]
        total_time = sum(param.τ for param in params)
        Ωf, θf = get_Ω_θ_func(params)
        seg_data, seg_grads = get_seg_data(params)
        SS.compute_single_mode!(result, seg_data, buffer)
        dis = PN.displacement(0, total_time, Ωf, θf, rtol=1e-8, atol=1e-8)
        cum_dis = PN.cumulative_displacement(0, total_time, Ωf, θf,
                                             rtol=1e-8, atol=1e-8)
        area = PN.enclosed_area(0, total_time, Ωf, θf, rtol=5e-6, atol=5e-6)
        @test result.val.τ ≈ total_time
        @test result.val.dis ≈ dis rtol=1e-3 atol=1e-3
        @test result.val.area ≈ area rtol=5e-2 atol=5e-2
        @test result.val.cumdis ≈ cum_dis rtol=1e-3 atol=1e-3
    end
    for i in 1:100
        test_nseg(1)
        test_nseg(2)
        test_nseg(5)
    end
end

function add_δ_offset(params, δ)
    times = accumulate(+, param.τ for param in params)
    pop!(times)
    insert!(times, 1, 0)
    return [SegParam(param.τ, param.Ω, param.Ω′, param.φ + time * δ, param.δ + δ)
            for (time, param) in zip(times, params)]
end
add_δ_offset_callback(params) = δ->add_δ_offset(params, δ)

function add_single_offset(params, idx; offsets...)
    function offseted_param(i)
        param = params[i]
        if i != idx
            return param
        end
        τ = :τ in keys(offsets) ? param.τ + values(offsets).τ : param.τ
        Ω = :Ω in keys(offsets) ? param.Ω + values(offsets).Ω : param.Ω
        Ω′ = :Ω′ in keys(offsets) ? param.Ω′ + values(offsets).Ω′ : param.Ω′
        φ = :φ in keys(offsets) ? param.φ + values(offsets).φ : param.φ
        δ = :δ in keys(offsets) ? param.δ + values(offsets).δ : param.δ
        return SegParam(τ, Ω, Ω′, φ, δ)
    end
    return [offseted_param(i) for i in 1:length(params)]
end
add_single_offset_callback(params, idx, name) =
    δ->add_single_offset(params, idx; name=>δ)

max_δ(params) = maximum(abs(param.δ) for param in params)
max_τ(params) = maximum(abs(param.τ) for param in params)

function compute_grad(v₋₄, v₋₃, v₋₂, v₋₁, v₁, v₂, v₃, v₄, h)
    return (-(v₄ - v₋₄) / 280 + 4 * (v₃ - v₋₃) / 105
            - (v₂ - v₋₂) / 5 + 4 * (v₁ - v₋₁) / 5) / h
end

@testset "Random sequence gradients" begin
    T = Float64
    mask_full = SS.ValueMask(true, true, true, true, true, true)
    mask_none = zero(SS.ValueMask)

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SingleModeResult{T}(Val(mask_full), Val(mask_full))

    all_params_array = [SegParam{T}(τ, Ω, Ω′, φ, δ) for (τ, Ω, Ω′, φ, δ) in all_params]

    function eval_params(result, params, need_grad=true)
        seg_data, seg_grads = get_seg_data(params)
        SS.compute_single_mode!(result, seg_data, buffer,
                                need_grad ? seg_grads : nothing)
        @test result.val.τ ≈ sum(param.τ for param in params)
        return
    end

    result′ = SS.SingleModeResult{T}(Val(mask_full), Val(mask_none))
    function eval_grad(param_cb, nh)
        function eval_wrapper(params)
            eval_params(result′, params, false)
            return result′.val
        end
        h = nh / 4
        hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
        results = eval_wrapper.(param_cb.(hs))
        return SS.SegData{T}(compute_grad((x->x.τ).(results)..., h),
                             compute_grad((x->x.dis).(results)..., h),
                             compute_grad((x->x.area).(results)..., h),
                             compute_grad((x->x.cumdis).(results)..., h),
                             compute_grad((x->x.disδ).(results)..., h),
                             compute_grad((x->x.areaδ).(results)..., h))
    end

    function test_nseg(nseg)
        params = [rand(all_params_array) for i in 1:nseg]
        eval_params(result, params)

        nhτ = 0.005 / max(max_δ(params), 1.0)
        nhΩ = 0.005
        nhΩ′ = 0.005
        nhδ = 0.005 / max(max_τ(params), 1.0)
        nhφ = 0.005

        grad_δ = eval_grad(add_δ_offset_callback(params), nhδ)

        grads = [[eval_grad(add_single_offset_callback(params, i, :τ), nhτ)
                  for i in 1:nseg],
                 [eval_grad(add_single_offset_callback(params, i, :Ω), nhΩ)
                  for i in 1:nseg],
                 [eval_grad(add_single_offset_callback(params, i, :Ω′), nhΩ′)
                  for i in 1:nseg],
                 [eval_grad(add_single_offset_callback(params, i, :φ), nhφ)
                  for i in 1:nseg],
                 [eval_grad(add_single_offset_callback(params, i, :δ), nhδ)
                  for i in 1:nseg]]

        @test result.val.disδ ≈ grad_δ.dis rtol=1e-8 atol=1e-8
        @test result.val.areaδ ≈ grad_δ.area rtol=1e-8 atol=1e-8

        for i in 1:nseg
            for j in 1:5
                @test(result.grad[i][j].τ ≈ grads[j][i].τ, rtol=1e-8, atol=1e-8)
                @test(result.grad[i][j].dis ≈ grads[j][i].dis,
                      rtol=1e-8, atol=1e-8)
                @test(result.grad[i][j].area ≈ grads[j][i].area,
                      rtol=1e-5, atol=1e-5)
                @test(result.grad[i][j].cumdis ≈ grads[j][i].cumdis,
                      rtol=1e-5, atol=1e-5)
                @test(result.grad[i][j].disδ ≈ grads[j][i].disδ,
                      rtol=1e-5, atol=1e-5)
                @test(result.grad[i][j].areaδ ≈ grads[j][i].areaδ,
                      rtol=1e-3, atol=1e-3)
            end
        end
    end
    for i in 1:1000
        test_nseg(1)
        test_nseg(2)
        test_nseg(5)
        test_nseg(10)
        test_nseg(20)
    end
end

end
