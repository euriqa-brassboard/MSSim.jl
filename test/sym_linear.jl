#!/usr/bin/julia

module SymLinear

using Test
using ForwardDiff
using Setfield
using Random

using MSSim
const PN = MSSim.PureNumeric
const SL = MSSim.SymLinear
const SS = MSSim.SegSeq

function get_Ω_θ_func(Ω, Ω′, φ, δ)
    return t->Ω + Ω′ * t, t->φ + δ * t
end

get_value(v) = v.value
get_value(v::Complex) = complex(get_value(real(v)), get_value(imag(v)))

get_partial(v, i) = v.partials[i]
get_partial(v::Complex, i) = complex(get_partial(real(v), i),
                                     get_partial(imag(v), i))

const τs = [0, 1, 5, 10, 20]
const Ωs = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0, 0.001, 0.02, 1, 5, 10, 20]
const Ω′s = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0,
               0.001, 0.02, 1, 5, 10, 20] ./ 10
const φs = range(0, 2π, 17)
const δs = [-20, -10, -5, -1, -0.1, -0.02, -0.001, 0,
             0.001, 0.02, 1, 5, 10, 20] ./ 10

const all_params = Iterators.product(τs, Ωs, Ω′s, φs, δs)

@testset "Displacement" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        Ωf, θf = get_Ω_θ_func(Ω, Ω′, φ, δ)
        v_num = PN.displacement(0, τ, Ωf, θf)
        v_sym = SL.SegInt.displacement(ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0)),
                                       ForwardDiff.Dual(Ω, (0.0, 1.0, 0.0, 0.0, 0.0)),
                                       ForwardDiff.Dual(Ω′, (0.0, 0.0, 1.0, 0.0, 0.0)),
                                       ForwardDiff.Dual(φ, (0.0, 0.0, 0.0, 1.0, 0.0)),
                                       ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 1.0)))
        v_sym_v = get_value(v_sym)
        v_sym_τ = get_partial(v_sym, 1)
        v_sym_Ω = get_partial(v_sym, 2)
        v_sym_Ω′ = get_partial(v_sym, 3)
        v_sym_φ = get_partial(v_sym, 4)
        v_sym_δ = get_partial(v_sym, 5)
        vgrad_sym = SL.SegInt.displacement_gradients(τ, Ω, Ω′, φ, δ)

        vδ_sym = SL.SegInt.displacement_δ(
            ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω, (0.0, 1.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω′, (0.0, 0.0, 1.0, 0.0, 0.0)),
            ForwardDiff.Dual(φ, (0.0, 0.0, 0.0, 1.0, 0.0)),
            ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 1.0)))
        vδ_sym_v = get_value(vδ_sym)
        vδ_sym_τ = get_partial(vδ_sym, 1)
        vδ_sym_Ω = get_partial(vδ_sym, 2)
        vδ_sym_Ω′ = get_partial(vδ_sym, 3)
        vδ_sym_φ = get_partial(vδ_sym, 4)
        vδ_sym_δ = get_partial(vδ_sym, 5)
        vδgrad_sym = SL.SegInt.displacement_δ_gradients(τ, Ω, Ω′, φ, δ)

        @test v_num ≈ v_sym_v atol = 1e-12 rtol = 1e-12
        @test v_sym_τ ≈ vgrad_sym[1] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω ≈ vgrad_sym[2] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω′ ≈ vgrad_sym[3] atol = 1e-12 rtol = 1e-12
        @test v_sym_φ ≈ vgrad_sym[4] atol = 1e-12 rtol = 1e-12
        @test v_sym_δ ≈ vgrad_sym[5] atol = 1e-12 rtol = 1e-12

        @test v_sym_δ ≈ vδ_sym_v atol = 1e-12 rtol = 1e-12
        @test vδ_sym_τ ≈ vδgrad_sym[1] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_Ω ≈ vδgrad_sym[2] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_Ω′ ≈ vδgrad_sym[3] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_φ ≈ vδgrad_sym[4] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_δ ≈ vδgrad_sym[5] atol = 1e-12 rtol = 1e-12
    end
end

@testset "Cumulative Displacement" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        Ωf, θf = get_Ω_θ_func(Ω, Ω′, φ, δ)
        v_num = PN.cumulative_displacement(0, τ, Ωf, θf)
        v_sym = SL.SegInt.cumulative_displacement(
            ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω, (0.0, 1.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω′, (0.0, 0.0, 1.0, 0.0, 0.0)),
            ForwardDiff.Dual(φ, (0.0, 0.0, 0.0, 1.0, 0.0)),
            ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 1.0)))
        v_sym_v = get_value(v_sym)
        v_sym_τ = get_partial(v_sym, 1)
        v_sym_Ω = get_partial(v_sym, 2)
        v_sym_Ω′ = get_partial(v_sym, 3)
        v_sym_φ = get_partial(v_sym, 4)
        v_sym_δ = get_partial(v_sym, 5)
        vgrad_sym = SL.SegInt.cumulative_displacement_gradients(τ, Ω, Ω′, φ, δ)

        @test v_num ≈ v_sym_v atol = 1e-12 rtol = 1e-12
        @test v_sym_τ ≈ vgrad_sym[1] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω ≈ vgrad_sym[2] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω′ ≈ vgrad_sym[3] atol = 1e-12 rtol = 1e-12
        @test v_sym_φ ≈ vgrad_sym[4] atol = 1e-12 rtol = 1e-12
        @test v_sym_δ ≈ vgrad_sym[5] atol = 1e-11 rtol = 1e-11
    end
end

@testset "Enclosed Area" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        Ωf, θf = get_Ω_θ_func(Ω, Ω′, φ, δ)
        v_num = PN.enclosed_area(0, τ, Ωf, θf)
        v_sym = SL.SegInt.enclosed_area(ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0)),
                                        ForwardDiff.Dual(Ω, (0.0, 1.0, 0.0, 0.0, 0.0)),
                                        ForwardDiff.Dual(Ω′, (0.0, 0.0, 1.0, 0.0, 0.0)),
                                        ForwardDiff.Dual(φ, (0.0, 0.0, 0.0, 1.0, 0.0)),
                                        ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 1.0)))
        v_sym_v = get_value(v_sym)
        v_sym_τ = get_partial(v_sym, 1)
        v_sym_Ω = get_partial(v_sym, 2)
        v_sym_Ω′ = get_partial(v_sym, 3)
        v_sym_φ = get_partial(v_sym, 4)
        v_sym_δ = get_partial(v_sym, 5)
        vgrad_sym = SL.SegInt.enclosed_area_gradients(τ, Ω, Ω′, φ, δ)

        vδ_sym = SL.SegInt.enclosed_area_δ(
            ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω, (0.0, 1.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω′, (0.0, 0.0, 1.0, 0.0, 0.0)),
            ForwardDiff.Dual(φ, (0.0, 0.0, 0.0, 1.0, 0.0)),
            ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 1.0)))
        vδ_sym_v = get_value(vδ_sym)
        vδ_sym_τ = get_partial(vδ_sym, 1)
        vδ_sym_Ω = get_partial(vδ_sym, 2)
        vδ_sym_Ω′ = get_partial(vδ_sym, 3)
        vδ_sym_φ = get_partial(vδ_sym, 4)
        vδ_sym_δ = get_partial(vδ_sym, 5)
        vδgrad_sym = SL.SegInt.enclosed_area_δ_gradients(τ, Ω, Ω′, φ, δ)

        @test v_num ≈ v_sym_v atol = 2e-11 rtol = 1e-11
        @test v_sym_τ ≈ vgrad_sym[1] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω ≈ vgrad_sym[2] atol = 1e-12 rtol = 1e-12
        @test v_sym_Ω′ ≈ vgrad_sym[3] atol = 1e-12 rtol = 1e-12
        @test v_sym_φ == vgrad_sym[4] == 0
        @test v_sym_δ ≈ vgrad_sym[5] atol = 1e-12 rtol = 1e-12

        @test v_sym_δ ≈ vδ_sym_v atol = 1e-12 rtol = 1e-12
        @test vδ_sym_τ ≈ vδgrad_sym[1] atol = 2e-11 rtol = 1e-11
        @test vδ_sym_Ω ≈ vδgrad_sym[2] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_Ω′ ≈ vδgrad_sym[3] atol = 1e-12 rtol = 1e-12
        @test vδ_sym_φ == vδgrad_sym[4] == 0
        @test vδ_sym_δ ≈ vδgrad_sym[5] atol = 1e-12 rtol = 1e-12

        vc_num = PN.enclosed_area_complex(0, τ, Ωf, θf)
        vc_sym = SL.SegInt.enclosed_area_complex(τ, Ω, Ω′, φ, δ)
        @test vc_num ≈ vc_sym atol = 1e-10 rtol = 1e-10
    end
end

@testset "Special cases" begin
    for (τ, Ω, φ) in Iterators.product(τs, Ωs, φs)
        @test SL.SegInt.displacement(τ, Ω, 0, φ, 0) ≈ τ * Ω * cis(φ)
        @test SL.SegInt.displacement_δ(τ, Ω, 0, φ, 0) ≈ im * τ^2 * Ω * cis(φ) / 2
        @test SL.SegInt.cumulative_displacement(τ, Ω, 0, φ, 0) ≈ τ^2 * Ω * cis(φ) / 2
        @test SL.SegInt.enclosed_area(τ, Ω, 0, φ, 0) == 0
        @test SL.SegInt.enclosed_area_δ(τ, Ω, 0, φ, 0) ≈ abs2(τ * Ω) * τ / 6
    end
end

@testset "Compute values" begin
    for (τ, Ω, Ω′, φ, δ) in all_params
        v_dis = SL.SegInt.displacement(τ, Ω, Ω′, φ, δ)
        v_disδ = SL.SegInt.displacement_δ(τ, Ω, Ω′, φ, δ)
        v_dis_grad = SL.SegInt.displacement_gradients(τ, Ω, Ω′, φ, δ)
        v_disδ_grad = SL.SegInt.displacement_δ_gradients(τ, Ω, Ω′, φ, δ)
        v_cumdis = SL.SegInt.cumulative_displacement(τ, Ω, Ω′, φ, δ)
        v_cumdis_grad = SL.SegInt.cumulative_displacement_gradients(τ, Ω, Ω′, φ, δ)
        v_area = SL.SegInt.enclosed_area(τ, Ω, Ω′, φ, δ)
        v_areaδ = SL.SegInt.enclosed_area_δ(τ, Ω, Ω′, φ, δ)
        v_area_grad = SL.SegInt.enclosed_area_gradients(τ, Ω, Ω′, φ, δ)
        v_areaδ_grad = SL.SegInt.enclosed_area_δ_gradients(τ, Ω, Ω′, φ, δ)

        for (need_cumdis, need_area_mode, need_grad) in
            Iterators.product((false, true), (false, true), (false, true))

            maskv = SS.ValueMask(true, true, true, need_cumdis,
                                 need_area_mode, need_area_mode)
            maskg = need_grad ? maskv : zero(SS.ValueMask)

            d, grad = SL.SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(maskv), Val(maskg))
            @test d.dis ≈ v_dis atol=1e-10 rtol=1e-10
            @test d.area ≈ v_area atol=1e-10 rtol=1e-10
            if need_cumdis
                @test d.cumdis ≈ v_cumdis atol=1e-10 rtol=1e-10
            end
            if need_area_mode
                @test d.disδ ≈ v_disδ atol=1e-10 rtol=1e-10
                @test d.areaδ ≈ v_areaδ atol=1e-10 rtol=1e-10
            end
            if need_grad
                @test length(grad) == 5

                @test grad[1].dis ≈ v_dis_grad[1] atol=1e-10 rtol=1e-10
                @test grad[2].dis ≈ v_dis_grad[2] atol=1e-10 rtol=1e-10
                @test grad[3].dis ≈ v_dis_grad[3] atol=1e-10 rtol=1e-10
                @test grad[4].dis ≈ v_dis_grad[4] atol=1e-10 rtol=1e-10
                @test grad[5].dis ≈ v_dis_grad[5] atol=1e-10 rtol=1e-10
                @test grad[1].area ≈ v_area_grad[1] atol=1e-10 rtol=1e-10
                @test grad[2].area ≈ v_area_grad[2] atol=1e-10 rtol=1e-10
                @test grad[3].area ≈ v_area_grad[3] atol=1e-10 rtol=1e-10
                @test grad[4].area ≈ v_area_grad[4] atol=1e-10 rtol=1e-10
                @test grad[5].area ≈ v_area_grad[5] atol=1e-10 rtol=1e-10

                if need_cumdis
                    @test grad[1].cumdis ≈ v_cumdis_grad[1] atol=1e-10 rtol=1e-10
                    @test grad[2].cumdis ≈ v_cumdis_grad[2] atol=1e-10 rtol=1e-10
                    @test grad[3].cumdis ≈ v_cumdis_grad[3] atol=1e-10 rtol=1e-10
                    @test grad[4].cumdis ≈ v_cumdis_grad[4] atol=1e-10 rtol=1e-10
                    @test grad[5].cumdis ≈ v_cumdis_grad[5] atol=1e-10 rtol=1e-10
                end

                if need_area_mode
                    @test grad[1].disδ ≈ v_disδ_grad[1] atol=1e-10 rtol=1e-10
                    @test grad[2].disδ ≈ v_disδ_grad[2] atol=1e-10 rtol=1e-10
                    @test grad[3].disδ ≈ v_disδ_grad[3] atol=1e-10 rtol=1e-10
                    @test grad[4].disδ ≈ v_disδ_grad[4] atol=1e-10 rtol=1e-10
                    @test grad[5].disδ ≈ v_disδ_grad[5] atol=1e-10 rtol=1e-10
                    @test grad[1].areaδ ≈ v_areaδ_grad[1] atol=1e-10 rtol=1e-10
                    @test grad[2].areaδ ≈ v_areaδ_grad[2] atol=1e-10 rtol=1e-10
                    @test grad[3].areaδ ≈ v_areaδ_grad[3] atol=1e-10 rtol=1e-10
                    @test grad[4].areaδ ≈ v_areaδ_grad[4] atol=1e-10 rtol=1e-10
                    @test grad[5].areaδ ≈ v_areaδ_grad[5] atol=1e-10 rtol=1e-10
                end
            else
                @test grad === nothing
            end
        end
    end
end

function compute_grad(v₋₄, v₋₃, v₋₂, v₋₁, v₁, v₂, v₃, v₄, h)
    return (-(v₄ - v₋₄) / 280 + 4 * (v₃ - v₋₃) / 105
            - (v₂ - v₋₂) / 5 + 4 * (v₁ - v₋₁) / 5) / h
end

@testset "Random single-mode sequence" begin
    function test_grad(grad1, grad2, τ, Ω, Ω′, φ, δ)
        nargs = length(grad1)
        for i in 1:5:nargs
            if τ
                @test grad1[i] ≈ grad2[i]
            end
            if Ω
                @test grad1[i + 1] ≈ grad2[i + 1]
            end
            if Ω′
                @test grad1[i + 2] ≈ grad2[i + 2]
            end
            if φ
                @test grad1[i + 3] ≈ grad2[i + 3]
            end
            if δ
                @test grad1[i + 4] ≈ grad2[i + 4]
            end
        end
    end
    function test_nseg(::Val{NSeg}) where NSeg
        full_buf = SL.ComputeBuffer{NSeg,Float64}(
            Val(SS.ValueMask(true, true, true, true, true, true)),
            Val(SS.ValueMask(true, true, true, true, true, true)))
        full_kern = SL.Kernel(
            full_buf, Val(SL.ParamGradMask(true, true, true, true, true)))
        args = Vector{Float64}(undef, NSeg * 5)
        args2 = Vector{Float64}(undef, NSeg * 5)
        grad_buff = Vector{Float64}(undef, NSeg * 5)
        grad_buff2 = Vector{Float64}(undef, NSeg * 5)

        function check_grad(valf, gradf)
            nargs = NSeg * 5
            args2 .= args
            gradf(grad_buff, full_kern, args...)
            h = 0.000005 / 4
            hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
            function eval_wrapper(i, d)
                args2[i] = args[i] + d
                return valf(full_kern, args2...)
            end
            for i in 1:nargs
                results = eval_wrapper.(i, hs)
                @test grad_buff[i] ≈ compute_grad(results..., h) rtol=1.5e-2 atol=5e-4
            end
        end

        for _ in 1:100
            rand!(args)
            rdis = SL.value_rdis(full_kern, args...)
            idis = SL.value_idis(full_kern, args...)
            area = SL.value_area(full_kern, args...)
            rcumdis = SL.value_rcumdis(full_kern, args...)
            icumdis = SL.value_icumdis(full_kern, args...)
            rdisδ = SL.value_rdisδ(full_kern, args...)
            idisδ = SL.value_idisδ(full_kern, args...)
            areaδ = SL.value_areaδ(full_kern, args...)

            check_grad(SL.value_rdis, SL.grad_rdis)
            check_grad(SL.value_idis, SL.grad_idis)
            check_grad(SL.value_area, SL.grad_area)
            check_grad(SL.value_rcumdis, SL.grad_rcumdis)
            check_grad(SL.value_icumdis, SL.grad_icumdis)
            check_grad(SL.value_rdisδ, SL.grad_rdisδ)
            check_grad(SL.value_idisδ, SL.grad_idisδ)
            check_grad(SL.value_areaδ, SL.grad_areaδ)

            for (τ, Ω, Ω′, φ, δ) in Iterators.product((false, true), (false, true),
                                                            (false, true), (false, true),
                                                            (false, true))
                kern = SL.Kernel(full_buf, Val(SL.ParamGradMask(τ, Ω, Ω′, φ, δ)))
                function check_param(val, valf, gradf)
                    @test valf(kern, args...) ≈ val
                    gradf(grad_buff, full_kern, args...)
                    gradf(grad_buff2, kern, args...)
                    test_grad(grad_buff, grad_buff2, τ, Ω, Ω′, φ, δ)
                end
                check_param(rdis, SL.value_rdis, SL.grad_rdis)
                check_param(idis, SL.value_idis, SL.grad_idis)
                check_param(area, SL.value_area, SL.grad_area)
                check_param(rcumdis, SL.value_rcumdis, SL.grad_rcumdis)
                check_param(icumdis, SL.value_icumdis, SL.grad_icumdis)
                check_param(rdisδ, SL.value_rdisδ, SL.grad_rdisδ)
                check_param(idisδ, SL.value_idisδ, SL.grad_idisδ)
                check_param(areaδ, SL.value_areaδ, SL.grad_areaδ)
            end
        end
    end
    test_nseg(Val(1))
    test_nseg(Val(2))
    test_nseg(Val(5))
    test_nseg(Val(10))
    test_nseg(Val(20))
end

end
