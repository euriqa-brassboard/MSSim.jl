#!/usr/bin/julia

module TestOptimizers

using MSSim
const U = MSSim.Utils
const SL = MSSim.SymLinear
const SS = MSSim.SegSeq
const Opts = MSSim.Optimizers

using Test
using Random
using LinearAlgebra

function compute_grad(v₋₄, v₋₃, v₋₂, v₋₁, v₁, v₂, v₃, v₄, h)
    return (-(v₄ - v₋₄) / 280 + 4 * (v₃ - v₋₃) / 105
            - (v₂ - v₋₂) / 5 + 4 * (v₁ - v₋₁) / 5) / h
end

function test_msparams_grad(params::Opts.MSParams{NSeg}, nrounds) where NSeg
    nuser = Opts.nparams(params)
    nraw = NSeg * 5

    grads_raw = Vector{Float64}(undef, nraw)
    grads_user = Vector{Float64}(undef, nuser)
    args_raw = Vector{Float64}(undef, nraw)
    args_user = Vector{Float64}(undef, nuser)
    args_user2 = Vector{Float64}(undef, nuser)

    for i in 1:nrounds
        rand!(grads_raw)
        rand!(args_user)
        args_user[1] += 0.1
        Opts.transform_argument(params, args_raw, args_user)
        v0 = dot(args_raw, grads_raw)
        Opts.transform_gradient(params, grads_user, grads_raw, args_raw, args_user)
        for ai in 1:nuser
            args_user2 .= args_user
            function eval_at(x)
                args_user2[ai] = args_user[ai] + x
                Opts.transform_argument(params, args_raw, args_user2)
                return dot(args_raw, grads_raw)
            end
            h = 0.0001 / 4
            hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
            gn = compute_grad(eval_at.(hs)..., h)
            @test gn ≈ grads_user[ai] rtol=1e-4 atol=1e-9
        end
    end
end

function test_msparams_args(params::Opts.MSParams{NSeg,NAmp},
                            args_raw, args_user) where {NSeg,NAmp}
    total_t = Opts.transform_argument(params, args_raw, args_user)
    τ = args_user[1]
    @test total_t ≈ NSeg * τ
    amps = [begin
                a = 0.0
                for ai in 1:NAmp
                    a += params.amps[ai][i] * args_user[ai + 1]
                end
                a
            end for i in 1:NSeg + 1]
    φ = 0.0
    for si in 1:NSeg
        @test args_raw[si * 5 - 4] == τ
        @test args_raw[si * 5 - 3] ≈ amps[si]
        @test args_raw[si * 5 - 2] ≈ (amps[si + 1] - amps[si]) / τ
        @test args_raw[si * 5 - 1] ≈ φ
        φ += args_raw[si * 5] * τ
    end
end

@testset "Parameter transform" begin
    for nseg in (1, 2, 5, 10)
        nraw = nseg * 5
        args_raw = Vector{Float64}(undef, nraw)
        for namp in (1, 2, 5)
            for _ in 1:100
                # No FM
                params = Opts.MSParams{nseg,namp,false,false}(
                    ntuple(_->rand(nseg + 1), namp))
                nuser = Opts.nparams(params)
                @test nuser == 2 + namp

                args_user = rand(nuser)
                args_user[1] += 0.1
                test_msparams_args(params, args_raw, args_user)

                ω = args_user[end]
                for si in 1:nseg
                    @test args_raw[si * 5] == ω
                end

                test_msparams_grad(params, 100)

                # FM
                params = Opts.MSParams{nseg,namp,false,true}(
                    ntuple(_->rand(nseg + 1), namp))
                nuser = Opts.nparams(params)
                @test nuser == 1 + namp + nseg

                args_user = rand(nuser)
                args_user[1] += 0.1
                test_msparams_args(params, args_raw, args_user)

                for si in 1:nseg
                    @test args_raw[si * 5] == args_user[1 + namp + si]
                end

                test_msparams_grad(params, 100)

                # Symmetric FM
                params = Opts.MSParams{nseg,namp,true,true}(
                    ntuple(_->rand(nseg + 1), namp))
                nuser = Opts.nparams(params)
                @test nuser == 1 + namp + (nseg + 1) ÷ 2

                args_user = rand(nuser)
                args_user[1] += 0.1
                test_msparams_args(params, args_raw, args_user)

                for si in 1:(nseg + 1) ÷ 2
                    @test args_raw[si * 5] == args_user[1 + namp + si]
                    @test args_raw[(nseg + 1 - si) * 5] == args_user[1 + namp + si]
                end

                test_msparams_grad(params, 100)
            end
        end
    end
end

end
