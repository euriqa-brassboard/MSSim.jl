#!/usr/bin/julia

using Test

using MSSim
const SL = MSSim.SymLinear
const SS = MSSim.SegSeq
const Seq = MSSim.Sequence

using Combinatorics
using ProtoBuf

function compute_grad(v₋₄, v₋₃, v₋₂, v₋₁, v₁, v₂, v₃, v₄, h)
    return (-(v₄ - v₋₄) / 280 + 4 * (v₃ - v₋₃) / 105
            - (v₂ - v₋₂) / 5 + 4 * (v₁ - v₋₁) / 5) / h
end

@testset "Objective" begin
    modes1 = Seq.Modes()
    push!(modes1, 2.5, 0.5)
    value_record = Ref(0.0)
    function objfunc1(vals, grads)
        grads[1] = 1
        value_record[] = vals[1]
        return vals[1]
    end

    modes3 = Seq.Modes()
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
        summarizer = Seq.Summarizer{nseg}()
        buf = SL.ComputeBuffer{nseg,Float64}(Val(SS.mask_full), Val(SS.mask_full))
        kern = SL.Kernel(buf, Val(SL.pmask_full))
        freq_spec = Seq.FreqSpec(true, sym=false)
        if amp_order == 0
            amp_spec = Seq.AmpSpec()
        else
            amp_spec = Seq.AmpSpec(cb=ntuple(i->(x->x^(i - 1)), amp_order + 1), sym=false)
        end
        param0 = Seq.ModSpec{nseg}(freq=freq_spec, amp=amp_spec)
        args_raw = Vector{Float64}(undef, nseg * 5)
        for _ in 1:60
            args_user = rand(Seq.nparams(param0))
            args_user[1] += 0.1 # τ
            args_user2 = similar(args_user)
            Seq.transform_argument(param0, args_raw, args_user)
            raw_params = Seq.RawParams(args_raw)
            SL.update!(kern, (get(raw_params; ωm=modes1[1][1])...,))
            function eval_model1(name, idx)
                model = Seq.Objective(SL.pmask_full, ((name, idx),),
                                      objfunc1, modes1, buf,
                                      freq=freq_spec, amp=amp_spec)
                @test Seq.nparams(model) == Seq.nparams(param0)
                grads = similar(args_user)
                res = model(args_user, grads)
                @test res == value_record[]
                @test model.args == args_raw
                @test Seq.RawParams(model, args_user).args == args_raw

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

                executed = false
                @test nothing === model(Val(()), args_user) do x
                    @test length(x) == 0
                    executed = true
                    return
                end
                @test executed
                @test nseg * args_user[1] ≈ model(Val((:τ, 0)), args_user) do x
                    @test length(x) == 1
                    return x[1]
                end
                @test "executed" == model(Val(((:rdis, 1), (:idis, 1), (:area, 1))), args_user) do x
                    @test length(x) == 3
                    @test x[1] ≈ real(kern.result.val.dis)
                    @test x[2] ≈ imag(kern.result.val.dis)
                    @test x[3] ≈ kern.result.val.area
                    return "executed"
                end

                for ai in 1:length(args_user)
                    args_user2 .= args_user
                    function eval_at(x)
                        args_user2[ai] = args_user[ai] + x
                        return model(args_user2, Float64[])
                    end
                    h = 0.0001 / 4
                    hs = (-4, -3, -2, -1, 1, 2, 3, 4) .* h
                    gn = compute_grad(eval_at.(hs)..., h)
                    @test gn ≈ grads[ai] rtol=1e-4 atol=1e-7
                end
                return res
            end

            solprops = get(summarizer, raw_params, modes1)
            @test solprops.total_time ≈ nseg * args_user[1]
            @test solprops.modes[1] == modes1.modes[1][1]
            @test solprops.dis[1] ≈ kern.result.val.dis
            @test solprops.disδ[1] ≈ kern.result.val.disδ
            @test solprops.cumdis[1] ≈ kern.result.val.cumdis
            @test solprops.area[1] ≈ kern.result.val.area
            @test solprops.areaδ[1] ≈ kern.result.val.areaδ

            solprops_data = Dict(solprops)
            solprops2 = Seq.SolutionProperties(solprops_data)
            @test solprops2.total_time == solprops.total_time
            @test solprops2.modes == solprops.modes
            @test solprops2.dis == solprops.dis
            @test solprops2.disδ == solprops.disδ
            @test solprops2.cumdis == solprops.cumdis
            @test solprops2.area == solprops.area
            @test solprops2.areaδ == solprops.areaδ

            solprops_io = IOBuffer()
            pb_encoder = ProtoEncoder(solprops_io)
            encode(pb_encoder, solprops)
            @test solprops_io.size == ProtoBuf._encoded_size(solprops)

            seekstart(solprops_io)
            pb_decoder = ProtoDecoder(solprops_io)
            solprops3 = decode(pb_decoder, Seq.SolutionProperties)
            @test solprops3.total_time == solprops.total_time
            @test solprops3.modes == solprops.modes
            @test solprops3.dis == solprops.dis
            @test solprops3.disδ == solprops.disδ
            @test solprops3.cumdis == solprops.cumdis
            @test solprops3.area == solprops.area
            @test solprops3.areaδ == solprops.areaδ

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

            solprops3 = get(summarizer, raw_params, modes3)
            @test solprops3.total_time ≈ nseg * args_user[1]
            for idx in 1:3
                SL.update!(kern, (get(raw_params; ωm=modes3[idx][1])...,))
                @test solprops3.modes[idx] == modes3.modes[idx][1]
                @test solprops3.dis[idx] ≈ kern.result.val.dis
                @test solprops3.disδ[idx] ≈ kern.result.val.disδ
                @test solprops3.cumdis[idx] ≈ kern.result.val.cumdis
                @test solprops3.area[idx] ≈ kern.result.val.area
                @test solprops3.areaδ[idx] ≈ kern.result.val.areaδ
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
            val_map[(:dis2, 0)] = Seq.total_dis(kern, raw_params, modes3)
            val_map[(:area, 0)] = Seq.total_area(kern, raw_params, modes3)
            val_map[(:disδ2, 0)] = Seq.total_disδ(kern, raw_params, modes3)
            val_map[(:areaδ, 0)] = Seq.total_areaδ(kern, raw_params, modes3)
            val_map[(:areaδ2, 0)] = Seq.all_areaδ(kern, raw_params, modes3)
            val_map[(:cumdis2, 0)] = Seq.total_cumdis(kern, raw_params, modes3)
            val_map[(:τ, 0)] = nseg * args_user[1]

            val_keys = collect(keys(val_map))
            key_combs = collect(Combinatorics.combinations(val_keys, 3))

            function eval_model3(key1, key2, key3)
                model = Seq.Objective(SL.pmask_full, (key1, key2, key3),
                                      objfunc3, modes3, buf,
                                      freq=freq_spec, amp=amp_spec)
                args3 = Seq.RawParams(model, args_user).args
                @test args3 == args_raw
                args3′ = Seq.RawParams(model, args_user, buff=args3).args
                @test args3′ === args3
                @test args3′ == args_raw
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
                    @test gn ≈ grads[ai] rtol=1e-4 atol=1e-7
                end
                return res
            end
            eval_model3(rand(key_combs)...)
        end
    end
end
