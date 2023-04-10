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
    CT = Complex{T}
    A = SS.AreaData{T}
    CD = SS.CumDisData{T,CT}
    AG = SS.AreaModeData{T,CT}

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SingleModeResult{T,A,CD,AG}()

    for (τ, Ω, Ω′, φ, δ) in all_params
        d, = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(true), Val(true), Val(false))
        SS.compute_sequence!(result, [d], buffer)
        @test result.τ == τ
        @test result.area.dis == d.area.dis
        @test result.area.area == d.area.area
        @test result.cumdis.cumdis == d.cumdis.cumdis
        @test result.area_mode.disδ == d.area_mode.disδ
        @test result.area_mode.areaδ == d.area_mode.areaδ

        for (need_cumdis, need_area_mode) in Iterators.product((false, true),
                                                               (false, true))
            CD′ = need_cumdis ? CD : SS.DummyCumDisData
            AG′ = need_area_mode ? AG : SS.DummyAreaModeData

            result′ = SS.SingleModeResult{T,A,CD′,AG′}()
            d′, = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(need_cumdis),
                                            Val(need_area_mode), Val(false))
            SS.compute_sequence!(result′, [d′], buffer)
            @test result′.τ == τ
            @test result′.area.dis == d.area.dis
            @test result′.area.area == d.area.area
            if need_cumdis
                @test result′.cumdis.cumdis == d.cumdis.cumdis
            end
            if need_area_mode
                @test result′.area_mode.disδ == d.area_mode.disδ
                @test result′.area_mode.areaδ == d.area_mode.areaδ
            end
        end
    end
end

@testset "Trivial Segment" begin
    T = Float64
    CT = Complex{T}
    A = SS.AreaData{T}
    CD = SS.CumDisData{T,CT}
    AG = SS.AreaModeData{T,CT}

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SingleModeResult{T,A,CD,AG}()

    for (τ, Ω, Ω′, φ, δ) in all_params
        d, = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(true), Val(true), Val(false))
        nd, = SL.SegInt.compute_values(τ, Ω + Ω′ * τ, -Ω′, φ + δ * τ + π, -δ,
                                       Val(true), Val(true), Val(false))
        @test nd.τ == d.τ
        @test nd.area.dis ≈ -d.area.dis
        @test nd.area.area ≈ -d.area.area
        @test nd.cumdis.cumdis ≈ d.cumdis.cumdis - d.area.dis * τ
        @test nd.area_mode.disδ ≈ d.area_mode.disδ - im * τ * d.area.dis
        @test nd.area_mode.areaδ ≈ d.area_mode.areaδ
        SS.compute_sequence!(result, [d, nd], buffer)
        @test result.τ ≈ 2 * τ
        @test result.area.dis ≈ 0 atol=1e-8
        @test result.area.area ≈ 0 atol=1e-8
        @test result.cumdis.cumdis ≈ 2 * d.cumdis.cumdis
        @test result.area_mode.disδ ≈ 2 * d.area_mode.disδ - 2 * im * τ * d.area.dis
        @test result.area_mode.areaδ ≈
            (2 * imag(d.area_mode.disδ * conj(d.area.dis))
             - 2 * τ * abs2(d.area.dis) + 2 * d.area_mode.areaδ)

        SS.compute_sequence!(result, [d, d], buffer)
        @test result.τ ≈ 2 * τ
        @test result.area.dis ≈ 2 * d.area.dis atol=1e-8
        @test result.area.area ≈ 2 * d.area.area atol=1e-8
        @test result.cumdis.cumdis ≈ 2 * d.cumdis.cumdis + τ * d.area.dis
        @test result.area_mode.disδ ≈ 2 * d.area_mode.disδ + im * τ * d.area.dis
        @test result.area_mode.areaδ ≈ τ * abs2(d.area.dis) + 2 * d.area_mode.areaδ

        for (need_cumdis, need_area_mode) in Iterators.product((false, true),
                                                               (false, true))
            CD′ = need_cumdis ? CD : SS.DummyCumDisData
            AG′ = need_area_mode ? AG : SS.DummyAreaModeData

            result′ = SS.SingleModeResult{T,A,CD′,AG′}()
            d′, = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(need_cumdis),
                                            Val(need_area_mode), Val(false))
            SS.compute_sequence!(result′, [d′, d′], buffer)
            @test result′.τ == 2 * τ
            @test result′.area.dis ≈ 2 * d.area.dis atol=1e-8
            @test result′.area.area ≈ 2 * d.area.area atol=1e-8
            if need_cumdis
                @test result′.cumdis.cumdis ≈ 2 * d.cumdis.cumdis + τ * d.area.dis
            end
            if need_area_mode
                @test result′.area_mode.disδ ≈
                    2 * d.area_mode.disδ + im * τ * d.area.dis
                @test result′.area_mode.areaδ ≈
                    τ * abs2(d.area.dis) + 2 * d.area_mode.areaδ
            end
        end

        for τ′ in τs
            d0, = SL.SegInt.compute_values(τ′, 0.0, 0.0, 0.0, 0.0,
                                           Val(true), Val(true), Val(false))
            SS.compute_sequence!(result, [d0, d], buffer)
            @test result.τ ≈ τ + τ′
            @test result.area.dis == d.area.dis
            @test result.area.area == d.area.area
            @test result.cumdis.cumdis == d.cumdis.cumdis
            @test result.area_mode.disδ ≈ d.area_mode.disδ + im * τ′ * d.area.dis
            @test result.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result, [d, d0], buffer)
            @test result.τ ≈ τ + τ′
            @test result.area.dis == d.area.dis
            @test result.area.area == d.area.area
            @test result.cumdis.cumdis ≈ d.cumdis.cumdis + τ′ * d.area.dis
            @test result.area_mode.disδ == d.area_mode.disδ
            @test result.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result, [d0, d0, d], buffer)
            @test result.τ ≈ τ + τ′ * 2
            @test result.area.dis == d.area.dis
            @test result.area.area == d.area.area
            @test result.cumdis.cumdis == d.cumdis.cumdis
            @test result.area_mode.disδ ≈ d.area_mode.disδ + 2 * im * τ′ * d.area.dis
            @test result.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result, [d0, d, d0], buffer)
            @test result.τ ≈ τ + τ′ * 2
            @test result.area.dis == d.area.dis
            @test result.area.area == d.area.area
            @test result.cumdis.cumdis ≈ d.cumdis.cumdis + τ′ * d.area.dis
            @test result.area_mode.disδ ≈ d.area_mode.disδ + im * τ′ * d.area.dis
            @test result.area_mode.areaδ == d.area_mode.areaδ

            SS.compute_sequence!(result, [d, d0, d0], buffer)
            @test result.τ ≈ τ + τ′ * 2
            @test result.area.dis == d.area.dis
            @test result.area.area == d.area.area
            @test result.cumdis.cumdis ≈ d.cumdis.cumdis + 2 * τ′ * d.area.dis
            @test result.area_mode.disδ == d.area_mode.disδ
            @test result.area_mode.areaδ == d.area_mode.areaδ
        end
    end
end

@testset "Average zero" begin
    T = Float64
    CT = Complex{T}
    A = SS.AreaData{T}
    CD = SS.CumDisData{T,CT}
    AG = SS.AreaModeData{T,CT}

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SingleModeResult{T,A,CD,AG}()

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
                                       Val(true), Val(true), Val(false))
        d2, = SL.SegInt.compute_values(τ, Ω * abs(l2), 0, φ + angle(l2), 0,
                                       Val(true), Val(true), Val(false))
        d3, = SL.SegInt.compute_values(τ, Ω * abs(l3), 0, φ + angle(l3), 0,
                                       Val(true), Val(true), Val(false))
        d4, = SL.SegInt.compute_values(τ, Ω * abs(l4), 0, φ + angle(l4), 0,
                                       Val(true), Val(true), Val(false))

        SS.compute_sequence!(result, [d1, d2, d3, d4], buffer)
        @test result.τ == 4 * τ
        @test result.area.dis ≈ 0 atol=1e-10
        @test result.area.area ≈ -(Ω * τ)^2 * 4
        @test result.cumdis.cumdis ≈ 0 atol=1e-10
        @test result.area_mode.disδ ≈ 0 atol=1e-10
        if τ != 0 && Ω != 0
            @test result.area_mode.areaδ != 0
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
    CT = Complex{T}
    A = SS.AreaData{T}
    CD = SS.CumDisData{T,CT}
    AG = SS.AreaModeData{T,CT}
    grads = U.JaggedMatrix{SS.SegData{T,A,CD,AG}}()
    function compute_values(param)
        res, grad = SL.SegInt.compute_values(param.τ, param.Ω, param.Ω′, param.φ,
                                             param.δ, Val(true), Val(true), Val(true))
        push!(grads, grad)
        return res
    end
    return [compute_values(param) for param in params], grads
end

@testset "Random sequence" begin
    T = Float64
    CT = Complex{T}
    A = SS.AreaData{T}
    CD = SS.CumDisData{T,CT}
    AG = SS.AreaModeData{T,CT}

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SingleModeResult{T,A,CD,AG}()

    all_params_array = [SegParam{T}(τ, Ω, Ω′, φ, δ) for (τ, Ω, Ω′, φ, δ) in all_params]

    function test_nseg(nseg)
        params = [rand(all_params_array) for i in 1:nseg]
        total_time = sum(param.τ for param in params)
        Ωf, θf = get_Ω_θ_func(params)
        seg_data, seg_grads = get_seg_data(params)
        SS.compute_sequence!(result, seg_data, buffer)
        dis = PN.displacement(0, total_time, Ωf, θf, rtol=1e-8, atol=1e-8)
        cum_dis = PN.cumulative_displacement(0, total_time, Ωf, θf,
                                             rtol=1e-8, atol=1e-8)
        area = PN.enclosed_area(0, total_time, Ωf, θf, rtol=5e-6, atol=5e-6)
        @test result.τ ≈ total_time
        @test result.area.dis ≈ dis rtol=1e-3 atol=1e-3
        @test result.area.area ≈ area rtol=3e-2 atol=3e-2
        @test result.cumdis.cumdis ≈ cum_dis rtol=1e-3 atol=1e-3
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
    CT = Complex{T}
    A = SS.AreaData{T}
    CD = SS.CumDisData{T,CT}
    AG = SS.AreaModeData{T,CT}

    buffer = SS.SeqComputeBuffer{T}()
    result = SS.SingleModeResult{T,A,CD,AG}()

    all_params_array = [SegParam{T}(τ, Ω, Ω′, φ, δ) for (τ, Ω, Ω′, φ, δ) in all_params]

    function eval_params(result, params, include_grad=true)
        seg_data, seg_grads = get_seg_data(params)
        SS.compute_sequence!(result, seg_data, buffer,
                             include_grad ? seg_grads : nothing)
        @test result.τ ≈ sum(param.τ for param in params)
        return
    end

    result′ = SS.SingleModeResult{T,A,CD,AG}()
    function eval_grad(param_cb, nh)
        function eval_wrapper(params)
            eval_params(result′, params, false)
            return SS.SegData(result′.τ, result′.area,
                              result′.cumdis, result′.area_mode)
        end
        h = nh / 4
        hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
        results = eval_wrapper.(param_cb.(hs))
        return SS.SegData(compute_grad((x->x.τ).(results)..., h),
                          A(compute_grad((x->x.area.dis).(results)..., h),
                            compute_grad((x->x.area.area).(results)..., h)),
                          CD(compute_grad((x->x.cumdis.cumdis).(results)..., h)),
                          AG(compute_grad((x->x.area_mode.disδ).(results)..., h),
                             compute_grad((x->x.area_mode.areaδ).(results)..., h)))
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

        @test result.area_mode.disδ ≈ grad_δ.area.dis rtol=1e-8 atol=1e-8
        @test result.area_mode.areaδ ≈ grad_δ.area.area rtol=1e-8 atol=1e-8

        for i in 1:nseg
            for j in 1:5
                @test(result.τ_grad[i][j] ≈ grads[j][i].τ, rtol=1e-8, atol=1e-8)
                @test(result.area_grad[i][j].dis ≈ grads[j][i].area.dis,
                      rtol=1e-8, atol=1e-8)
                @test(result.area_grad[i][j].area ≈ grads[j][i].area.area,
                      rtol=1e-5, atol=1e-5)
                @test(result.cumdis_grad[i][j].cumdis ≈ grads[j][i].cumdis.cumdis,
                      rtol=1e-5, atol=1e-5)
                @test(result.area_mode_grad[i][j].disδ ≈ grads[j][i].area_mode.disδ,
                      rtol=1e-5, atol=1e-5)
                @test(result.area_mode_grad[i][j].areaδ ≈ grads[j][i].area_mode.areaδ,
                      rtol=1e-4, atol=1e-4)
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
