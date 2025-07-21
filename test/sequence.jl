#!/usr/bin/julia

using Test

using MSSim
const SL = MSSim.SymLinear
const SS = MSSim.SegSeq
const Seq = MSSim.Sequence

using Random
using LinearAlgebra

function compute_grad(v₋₄, v₋₃, v₋₂, v₋₁, v₁, v₂, v₃, v₄, h)
    return (-(v₄ - v₋₄) / 280 + 4 * (v₃ - v₋₃) / 105
            - (v₂ - v₋₂) / 5 + 4 * (v₁ - v₋₁) / 5) / h
end

function test_msparams_grad(params::Seq.ModSpec{NSeg}, nrounds) where NSeg
    nuser = Seq.nparams(params)
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
        Seq.transform_argument(params, args_raw, args_user)
        v0 = dot(args_raw, grads_raw)
        Seq.transform_gradient(params, grads_user, grads_raw, args_raw, args_user)
        for ai in 1:nuser
            args_user2 .= args_user
            function eval_at(x)
                args_user2[ai] = args_user[ai] + x
                Seq.transform_argument(params, args_raw, args_user2)
                return dot(args_raw, grads_raw)
            end
            h = 0.0001 / 4
            hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
            gn = compute_grad(eval_at.(hs)..., h)
            @test gn ≈ grads_user[ai] rtol=1e-4 atol=1e-9
        end
    end
end

function test_msparams_args(params::Seq.ModSpec{NSeg,NAmp},
                            args_raw, args_user) where {NSeg,NAmp}
    total_t = Seq.transform_argument(params, args_raw, args_user)
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
            amp_spec = Seq.AmpSpec(cb=ntuple(_->(_->rand()), namp), sym=false)
            for _ in 1:100
                # No FM
                params = Seq.ModSpec{nseg}(amp=amp_spec,
                                           freq=Seq.FreqSpec(false, sym=false))
                nuser = Seq.nparams(params)
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
                params = Seq.ModSpec{nseg}(amp=amp_spec,
                                           freq=Seq.FreqSpec(true, sym=false))
                nuser = Seq.nparams(params)
                @test nuser == 1 + namp + nseg

                args_user = rand(nuser)
                args_user[1] += 0.1
                test_msparams_args(params, args_raw, args_user)

                for si in 1:nseg
                    @test args_raw[si * 5] == args_user[1 + namp + si]
                end

                test_msparams_grad(params, 100)

                # Symmetric FM
                params = Seq.ModSpec{nseg}(amp=amp_spec,
                                           freq=Seq.FreqSpec(true, sym=true))
                nuser = Seq.nparams(params)
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

@testset "RawParams" begin
    raw_params = Seq.RawParams(1:15)
    raw_params2 = Seq.adjust(raw_params; tmax=19)
    @test raw_params2.args == raw_params.args
    raw_params3 = Seq.adjust(raw_params; tmax=5)
    @test raw_params3.args[1:5] == raw_params.args[1:5]
    @test raw_params3.args[6] == 4
    @test raw_params3.args[7:10] == raw_params.args[7:10]
    @test raw_params3.args[11] == 0
    @test raw_params3.args[12:15] == raw_params.args[12:15]
    raw_params4 = Seq.adjust(raw_params; δ=-1)
    @test raw_params4.args == [1, 2, 3, 4, 4,
                               6, 7, 8, 8, 9,
                               11, 12, 13, 7, 14]
    raw_params5 = Seq.adjust(raw_params; ωm=-1)
    @test raw_params5.args == [1, 2, 3, 4, 6,
                               6, 7, 8, 10, 11,
                               11, 12, 13, 21, 16]
    raw_params6 = Seq.adjust(raw_params; ωm=1, δ=1)
    @test raw_params6.args == raw_params.args
    @test Seq.get_Ωs(raw_params) == ([0, 1, 1, 7, 7, 18], [2, 5, 7, 55, 12, 155])
    @test Seq.get_ωs(raw_params) == ([0, 1, 1, 7, 7, 18], [5, 5, 10, 10, 15, 15])
    @test Seq.get_φs(raw_params) == ([0, 1, 1, 7, 7, 18], [4, 9, 9, 69, 14, 179])
    @test Seq.get_φs(raw_params, δ=-2) == ([0, 1, 1, 7, 7, 18], [4, 7, 7, 55, 0, 143])

    gate_info = Seq.gate_solution_info(raw_params, normalize_amp=false)
    @test sort(collect(keys(gate_info))) == ["amp", "amp_slope", "nsteps",
                                             "phase", "phase_slope", "time"]
    @test gate_info["nsteps"] == 3
    @test gate_info["time"] == [1, 6, 11]
    @test gate_info["amp"] == [2, 7, 12]
    @test gate_info["amp_slope"] == [3, 8, 13]
    @test gate_info["phase"] ≈ [0.6366197723675814, 0.432394487827058, 0.228169203286535]
    @test gate_info["phase_slope"] ≈ [0.7957747154594768, 1.5915494309189535,
                                       2.3873241463784303]

    gate_info = Seq.gate_solution_info(raw_params)
    @test sort(collect(keys(gate_info))) == ["amp", "amp_slope", "nsteps",
                                             "phase", "phase_slope", "time"]
    @test gate_info["nsteps"] == 3
    @test gate_info["time"] == [1, 6, 11]
    @test gate_info["amp"] ≈ [2, 7, 12] ./ 155
    @test gate_info["amp_slope"] ≈ [3, 8, 13] ./ 155
    @test gate_info["phase"] ≈ [0.6366197723675814, 0.432394487827058, 0.228169203286535]
    @test gate_info["phase_slope"] ≈ [0.7957747154594768, 1.5915494309189535,
                                       2.3873241463784303]

    raw_params.args[3:5:end] .= .-raw_params.args[3:5:end]
    gate_info = Seq.gate_solution_info(raw_params, normalize_amp=true)
    @test sort(collect(keys(gate_info))) == ["amp", "amp_slope", "nsteps",
                                             "phase", "phase_slope", "time"]
    @test gate_info["nsteps"] == 3
    @test gate_info["time"] == [1, 6, 11]
    @test gate_info["amp"] ≈ [2, 7, 12] ./ 131
    @test gate_info["amp_slope"] ≈ [-3, -8, -13] ./ 131
    @test gate_info["phase"] ≈ [0.6366197723675814, 0.432394487827058, 0.228169203286535]
    @test gate_info["phase_slope"] ≈ [0.7957747154594768, 1.5915494309189535,
                                       2.3873241463784303]
end

@testset "Trajectory" begin
    raw_params = Seq.RawParams(rand(15))
    ts, xs, ys = Seq.get_trajectory(raw_params, 10001)
    @test length(ts) == length(xs) == length(ys) == 10001
    buf = SL.ComputeBuffer{3,Float64}(Val(SS.ValueMask(true, true, false, false, false, false)), Val(zero(SS.ValueMask)))
    kern = SL.Kernel(buf, Val(SL.ParamGradMask(false, false, false, false, false)))
    for (t, x, y) in zip(ts, xs, ys)
        args = Seq.get(raw_params, tmax=t)
        @test x ≈ SL.value_rdis(kern, args...)
        @test y ≈ SL.value_idis(kern, args...)
    end
end
