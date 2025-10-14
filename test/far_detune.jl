#!/usr/bin/julia

using Test
using QuadGK
using ForwardDiff

using MSSim: FarDetune as FD

function numeric_area(τ, Ω11, Ω12, Ω21, Ω22, δ; atol=1e-8, rtol=1e-8)
    Ω1(t) = Ω11 + (Ω12 - Ω11) * (t / τ)
    Ω2(t) = Ω21 + (Ω22 - Ω21) * (t / τ)
    f(t) = Ω1(t) * Ω2(t) / δ / 2
    res, err = quadgk(f, 0, τ; atol=atol, rtol=rtol)
    return res
end

@testset "FarDetune area kernel" begin
    ωs0 = zeros(1)
    ωs0_2 = zeros(2)
    weights0 = ones(1)
    weights0_2 = ones(2)

    grad = zeros(6)
    grad2 = zeros(6)
    grad3 = zeros(6)
    for _ in 1:200
        τ = rand()
        Ω11 = rand() - 0.5
        Ω12 = rand() - 0.5
        Ω21 = rand() - 0.5
        Ω22 = rand() - 0.5
        δ = rand() + 0.5

        vn = numeric_area(τ, Ω11, Ω12, Ω21, Ω22, δ)
        vs = FD.enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ, ())
        @test vs ≈ vn

        vs_autodiff = FD.enclosed_area_kernel(
            ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω11, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω12, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω21, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω22, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
            ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)), ())
        @test vs_autodiff.value ≈ vs

        vs_grad = FD.enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ, grad)
        @test vs_grad ≈ vs
        @test grad ≈ vs_autodiff.partials

        vs2_grad = FD.enclosed_area_modes(τ, Ω11, Ω12, Ω21, Ω22, δ, ωs0, weights0, grad2)
        @test vs2_grad ≈ vs
        @test grad2 ≈ grad

        grad2 .= 1
        vs0 = FD.enclosed_area_modes(τ, Ω11, Ω12, Ω21, Ω22, δ, Float64[],
                                     Float64[], grad2)
        @test vs0 == 0
        @test grad2 == zeros(6)

        grad2 .= 1
        vs2_grad = FD.enclosed_area_modes(τ, Ω11, Ω12, Ω21, Ω22, δ,
                                          ωs0_2, weights0_2, grad2)
        @test vs2_grad ≈ 2 * vs
        @test grad2 ≈ 2 .* grad

        ωs = rand(5) .- 0.8
        weights = rand(5) .- 0.5
        vs2 = 0.0
        for (ω, weight) in zip(ωs, weights)
            vs2 += FD.enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ - ω, ()) * weight
        end
        vs3 = FD.enclosed_area_modes(τ, Ω11, Ω12, Ω21, Ω22, δ, ωs, weights, ())
        @test vs3 ≈ vs2

        vs2_autodiff = FD.enclosed_area_modes(
            ForwardDiff.Dual(τ, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω11, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω12, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω21, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
            ForwardDiff.Dual(Ω22, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
            ForwardDiff.Dual(δ, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)), ωs, weights, ())
        @test vs2_autodiff.value ≈ vs2

        grad2 .= 1
        vs2_grad = FD.enclosed_area_modes(τ, Ω11, Ω12, Ω21, Ω22, δ, ωs, weights, grad2)
        @test vs2_grad ≈ vs2
        @test grad2 ≈ vs2_autodiff.partials

        vs_seq = FD.enclosed_area_seq([τ], [Ω11, Ω12], [Ω21, Ω22], [δ], ωs, weights,
                                      (), (), (), ())
        @test vs_seq ≈ vs2
        vs_seq2 = FD.enclosed_area_seq([τ], [Ω11, Ω12], [Ω21, Ω22], [δ], ωs, weights,
                                       @view(grad3[1:1]), @view(grad3[2:3]),
                                       @view(grad3[4:5]), @view(grad3[6:6]))
        @test vs_seq2 ≈ vs2
        @test grad3 ≈ grad2
    end

    grad4 = zeros(10 + 11 + 11 + 10)
    for _ in 1:100
        ωs = rand(5) .- 0.8
        weights = rand(5) .- 0.5

        τs = rand(10)
        Ω1s = rand(11) .- 0.5
        Ω2s = rand(11) .- 0.5
        δs = rand(10) .+ 0.5

        v1 = 0.0
        for i in 1:10
            v1 += FD.enclosed_area_modes(τs[i], Ω1s[i], Ω1s[i + 1], Ω2s[i], Ω2s[i + 1],
                                         δs[i], ωs, weights, ())
        end
        vseq = FD.enclosed_area_seq(τs, Ω1s, Ω2s, δs, ωs, weights, (), (), (), ())
        @test vseq ≈ v1

        τs_dual = [ForwardDiff.Dual(τs[i], ntuple(j->Float64(j == i), 42)) for i in 1:10]
        Ω1s_dual = [ForwardDiff.Dual(Ω1s[i], ntuple(j->Float64(j == i + 10), 42))
                     for i in 1:11]
        Ω2s_dual = [ForwardDiff.Dual(Ω2s[i], ntuple(j->Float64(j == i + 21), 42))
                     for i in 1:11]
        δs_dual = [ForwardDiff.Dual(δs[i], ntuple(j->Float64(j == i + 32), 42))
                    for i in 1:10]

        vseq_autodiff = FD.enclosed_area_seq(τs_dual, Ω1s_dual, Ω2s_dual, δs_dual,
                                             ωs, weights, (), (), (), ())
        @test vseq_autodiff.value ≈ v1
        vseq2 = FD.enclosed_area_seq(τs, Ω1s, Ω2s, δs, ωs, weights,
                                     @view(grad4[1:10]), @view(grad4[11:21]),
                                     @view(grad4[22:32]), @view(grad4[33:42]))
        @test vseq2 ≈ v1
        @test grad4 ≈ vseq_autodiff.partials
    end
end

@testset "FarDetune area target" begin
    NIons = 10
    NPairs = NIons * (NIons - 1) ÷ 2

    tgt = FD.AreaTargets{NIons}()
    @test length(tgt.targets) == NPairs
    @test length(tgt.weights) == NPairs
    @test all(tgt.targets .== 0)
    @test all(tgt.weights .== 1)

    index_map = Dict{NTuple{2,Int},Int}()
    pair_idx = 0
    for ion1 in 1:NIons - 1
        for ion2 in ion1 + 1:NIons
            pair_idx += 1
            index_map[(ion1, ion2)] = pair_idx
            index_map[(ion2, ion1)] = pair_idx
            @test FD.pair_idx(tgt, ion1, ion2) == pair_idx
            @test FD.pair_idx(tgt, ion2, ion1) == pair_idx
            @test FD.pair_idx(NIons, ion1, ion2) == pair_idx
            @test FD.pair_idx(NIons, ion2, ion1) == pair_idx
        end
    end
    @test_throws BoundsError tgt[0, 1]
    @test_throws BoundsError tgt[NIons + 1, 1]
    @test_throws BoundsError FD.getweight(tgt, 0, 1)
    @test_throws BoundsError FD.getweight(tgt, NIons + 1, 1)
    for ion1 in 1:NIons
        @test_throws BoundsError tgt[ion1, ion1]
        @test_throws BoundsError FD.getweight(tgt, ion1, ion1)
    end
    @test NPairs == pair_idx

    grad = zeros(NPairs)
    for _ in 1:100
        targets = rand(NPairs)
        weights = rand(NPairs)
        xs = rand(NPairs)
        pair_idx = 0
        for ion1 in 1:NIons - 1
            for ion2 in ion1 + 1:NIons
                pair_idx += 1
                t = targets[pair_idx]
                w = weights[pair_idx]

                tgt[ion1, ion2] = t
                @test tgt[ion1, ion2] == t
                @test tgt[ion2, ion1] == t
                @test tgt.targets[index_map[(ion1, ion2)]] == t

                FD.setweight!(tgt, w, ion1, ion2)
                @test FD.getweight(tgt, ion1, ion2) == w
                @test FD.getweight(tgt, ion2, ion1) == w
                @test tgt.weights[index_map[(ion1, ion2)]] == w
            end
        end

        v = sum((xs .- targets).^2 .* weights)
        @test tgt(xs, ()) ≈ v
        @test tgt(xs, grad) ≈ v

        xs_dual = [ForwardDiff.Dual(xs[i], ntuple(j->Float64(j == i), NPairs))
                   for i in 1:NPairs]
        v_dual = sum((xs_dual .- targets).^2 .* weights)
        @test v_dual.value ≈ v
        @test grad ≈ v_dual.partials
    end
end

function compute_grad(v₋₄, v₋₃, v₋₂, v₋₁, v₁, v₂, v₃, v₄, h)
    return (-(v₄ - v₋₄) / 280 + 4 * (v₃ - v₋₃) / 105
            - (v₂ - v₋₂) / 5 + 4 * (v₁ - v₋₁) / 5) / h
end

@testset "FarDetune Kernel" begin
    NSeg = 10
    NModes = 6
    NIons = 4
    NPairs = NIons * (NIons - 1) ÷ 2
    tgt = FD.AreaTargets{NIons}()

    obj_args = Vector{Float64}(undef, NPairs)
    arg_obj_res = 0.0
    function arg_obj(x, grad)
        obj_args .= x
        return arg_obj_res::Float64
    end

    grad = Vector{Float64}(undef, NSeg * 2 + (NSeg + 1) * NIons)
    for _ in 1:50
        ωs = rand(NModes) .- 0.8
        bij = rand(NModes, NIons) .- 0.5
        ηs = rand(NModes) .+ 0.01
        tgt.targets .= rand.()
        tgt.weights .= rand.()
        arg_obj_res = rand()

        τs = rand(NSeg) .+ 0.2
        all_Ωs = ntuple(_->rand(NSeg + 1) .+ 0.1, NIons)
        δs = rand(NSeg) .+ 0.5
        xs = [τs; all_Ωs...; δs]

        kern = FD.Kernel{NSeg}(ωs, bij, ηs)
        @test kern isa FD.Kernel{NSeg,NModes,NIons}
        for ion1 in 1:NIons - 1
            for ion2 in ion1 + 1:NIons
                @test kern.weights[:, FD.pair_idx(tgt, ion1, ion2)] == bij[:, ion1] .* bij[:, ion2] .* ηs.^2
            end
        end

        @test kern(arg_obj, xs, ()) == arg_obj_res
        for ion1 in 1:NIons - 1
            for ion2 in ion1 + 1:NIons
                pair_idx = FD.pair_idx(tgt, ion1, ion2)
                @test FD.enclosed_area_seq(τs, all_Ωs[ion1], all_Ωs[ion2], δs,
                                           ωs, kern.weights[:, pair_idx],
                                           (), (), (), ()) ≈ obj_args[pair_idx]
            end
        end
        v0 = tgt(obj_args, ())
        @test kern(tgt, xs, ()) ≈ v0
        @test kern(tgt, xs, grad) ≈ v0

        xs2 = copy(xs)
        for xi in 1:length(xs)
            xs2 .= xs
            function eval_at(x)
                xs2[xi] = xs[xi] + x
                return kern(tgt, xs2, ())
            end
            h = 0.001 / 4
            hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
            gn = compute_grad(eval_at.(hs)..., h)
            @test gn ≈ grad[xi] rtol=1e-7 atol=1e-7
        end
    end
end
