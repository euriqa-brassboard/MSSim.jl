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
using Combinatorics

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

@testset "NL model" begin
    modes1 = Opts.Modes()
    push!(modes1, 2.5, 0.5)
    value_record = Ref(0.0)
    function objfunc1(vals, grads)
        grads[1] = 1
        value_record[] = vals[1]
        return vals[1]
    end

    modes3 = Opts.Modes()
    push!(modes3, 2.1, -0.3)
    push!(modes3, 2.5, 0.5)
    push!(modes3, 2.9, 2.0)
    function objfunc3(vals, grads)
        grads[1] = 0.9
        grads[2] = -0.2
        grads[3] = 2 * vals[3]
        return vals[1] * 0.9 - vals[2] * 0.2 + vals[3]^2
    end

    for (nseg, amp_order) in Iterators.product((1, 2, 5, 10), (0, 2, 5))
        buf = SL.ComputeBuffer{nseg,Float64}(Val(Opts.mask_full), Val(Opts.mask_full))
        kern = SL.Kernel(buf, Val(Opts.pmask_full))
        freq_spec = Opts.FreqSpec(true, sym=false)
        amp_spec = Opts.AmpSpec(mid_order=amp_order, sym=false)
        param0 = Opts.MSParams{nseg}(freq=freq_spec, amp=amp_spec)
        args_raw = Vector{Float64}(undef, nseg * 5)
        for _ in 1:100
            args_user = rand(Opts.nparams(param0))
            args_user[1] += 0.1 # τ
            args_user2 = similar(args_user)
            Opts.transform_argument(param0, args_raw, args_user)
            args_value = Opts.ArgsValue(args_raw)
            SL.update!(kern, (Opts.get_args(args_value, modes1)[1]...,))
            function eval_model1(name, idx)
                model = Opts.MSObjective(Opts.pmask_full, ((name, idx),),
                                         objfunc1, modes1, buf,
                                         freq=freq_spec, amp=amp_spec)
                grads = similar(args_user)
                res = model(args_user, grads)
                @test res == value_record[]
                @test model.args == args_raw
                @test Opts.get_args(model, args_user) == args_raw

                @test model(Val((:rdis, 1)), args_user) ≈ real(kern.result.val.dis)
                @test model(Val((:idis, 1)), args_user) ≈ imag(kern.result.val.dis)
                @test model(Val((:dis2, 1)), args_user) ≈ abs2(kern.result.val.dis)
                @test model(Val((:area, 1)), args_user) ≈ kern.result.val.area
                @test model(Val((:rdisδ, 1)), args_user) ≈ real(kern.result.val.disδ)
                @test model(Val((:idisδ, 1)), args_user) ≈ imag(kern.result.val.disδ)
                @test model(Val((:disδ2, 1)), args_user) ≈ abs2(kern.result.val.disδ)
                @test model(Val((:areaδ, 1)), args_user) ≈ kern.result.val.areaδ
                @test model(Val((:areaδ2, 1)), args_user) ≈ abs2(kern.result.val.areaδ)
                @test model(Val((:rcumdis, 1)), args_user) ≈ real(kern.result.val.cumdis)
                @test model(Val((:icumdis, 1)), args_user) ≈ imag(kern.result.val.cumdis)
                @test model(Val((:cumdis2, 1)), args_user) ≈ abs2(kern.result.val.cumdis)

                @test model(Val((:dis2, 0)), args_user) ≈ abs2(kern.result.val.dis)
                @test model(Val((:area, 0)), args_user) ≈ kern.result.val.area * 0.5
                @test model(Val((:disδ2, 0)), args_user) ≈ abs2(kern.result.val.disδ)
                @test model(Val((:areaδ, 0)), args_user) ≈ kern.result.val.areaδ * 0.5
                @test model(Val((:areaδ2, 0)), args_user) ≈ abs2(kern.result.val.areaδ)
                @test model(Val((:cumdis2, 0)), args_user) ≈ abs2(kern.result.val.cumdis)
                @test model(Val((:τ, 0)), args_user) ≈ nseg * args_user[1]

                for ai in 1:length(args_user)
                    args_user2 .= args_user
                    function eval_at(x)
                        args_user2[ai] = args_user[ai] + x
                        return model(args_user2, Float64[])
                    end
                    h = 0.0001 / 4
                    hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
                    gn = compute_grad(eval_at.(hs)..., h)
                    @test gn ≈ grads[ai] rtol=1e-4 atol=1e-9
                end
                return res
            end
            @test eval_model1(:rdis, 1) ≈ real(kern.result.val.dis)
            @test eval_model1(:idis, 1) ≈ imag(kern.result.val.dis)
            @test eval_model1(:dis2, 1) ≈ abs2(kern.result.val.dis)
            @test eval_model1(:area, 1) ≈ kern.result.val.area
            @test eval_model1(:rdisδ, 1) ≈ real(kern.result.val.disδ)
            @test eval_model1(:idisδ, 1) ≈ imag(kern.result.val.disδ)
            @test eval_model1(:disδ2, 1) ≈ abs2(kern.result.val.disδ)
            @test eval_model1(:areaδ, 1) ≈ kern.result.val.areaδ
            @test eval_model1(:areaδ2, 1) ≈ abs2(kern.result.val.areaδ)
            @test eval_model1(:rcumdis, 1) ≈ real(kern.result.val.cumdis)
            @test eval_model1(:icumdis, 1) ≈ imag(kern.result.val.cumdis)
            @test eval_model1(:cumdis2, 1) ≈ abs2(kern.result.val.cumdis)

            @test eval_model1(:dis2, 0) ≈ abs2(kern.result.val.dis)
            @test eval_model1(:area, 0) ≈ kern.result.val.area * 0.5
            @test eval_model1(:disδ2, 0) ≈ abs2(kern.result.val.disδ)
            @test eval_model1(:areaδ, 0) ≈ kern.result.val.areaδ * 0.5
            @test eval_model1(:areaδ2, 0) ≈ abs2(kern.result.val.areaδ)
            @test eval_model1(:cumdis2, 0) ≈ abs2(kern.result.val.cumdis)
            @test eval_model1(:τ, 0) ≈ nseg * args_user[1]

            val_map = Dict{Tuple{Symbol,Int},Float64}()
            for idx in 1:3
                SL.update!(kern, (Opts.get_args(args_value, modes3)[idx]...,))
                val_map[(:rdis, idx)] = real(kern.result.val.dis)
                val_map[(:idis, idx)] = imag(kern.result.val.dis)
                val_map[(:dis2, idx)] = abs2(kern.result.val.dis)
                val_map[(:area, idx)] = kern.result.val.area
                val_map[(:rdisδ, idx)] = real(kern.result.val.disδ)
                val_map[(:idisδ, idx)] = imag(kern.result.val.disδ)
                val_map[(:disδ2, idx)] = abs2(kern.result.val.disδ)
                val_map[(:areaδ, idx)] = kern.result.val.areaδ
                val_map[(:areaδ2, idx)] = abs2(kern.result.val.areaδ)
                val_map[(:rcumdis, idx)] = real(kern.result.val.cumdis)
                val_map[(:icumdis, idx)] = imag(kern.result.val.cumdis)
                val_map[(:cumdis2, idx)] = abs2(kern.result.val.cumdis)
            end
            val_map[(:dis2, 0)] = Opts.total_dis(kern, args_value, modes3)
            val_map[(:area, 0)] = Opts.total_area(kern, args_value, modes3)
            val_map[(:disδ2, 0)] = Opts.total_disδ(kern, args_value, modes3)
            val_map[(:areaδ, 0)] = Opts.total_areaδ(kern, args_value, modes3)
            val_map[(:areaδ2, 0)] = Opts.all_areaδ(kern, args_value, modes3)
            val_map[(:cumdis2, 0)] = Opts.total_cumdis(kern, args_value, modes3)
            val_map[(:τ, 0)] = nseg * args_user[1]

            val_keys = collect(keys(val_map))
            key_combs = collect(Combinatorics.combinations(val_keys, 3))

            function eval_model3(key1, key2, key3)
                model = Opts.MSObjective(Opts.pmask_full, (key1, key2, key3),
                                         objfunc3, modes3, buf,
                                         freq=freq_spec, amp=amp_spec)
                @test Opts.get_args(model, args_user) == args_raw
                grads = similar(args_user)
                res = model(args_user, grads)
                @test res ≈ val_map[key1] * 0.9 - val_map[key2] * 0.2 + val_map[key3]^2
                @test model.args == args_raw

                for ai in 1:length(args_user)
                    args_user2 .= args_user
                    function eval_at(x)
                        args_user2[ai] = args_user[ai] + x
                        return model(args_user2, Float64[])
                    end
                    h = 0.0001 / 4
                    hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
                    gn = compute_grad(eval_at.(hs)..., h)
                    @test gn ≈ grads[ai] rtol=1e-4 atol=1e-9
                end
                return res
            end
            eval_model3(rand(key_combs)...)
        end
    end
end

end
