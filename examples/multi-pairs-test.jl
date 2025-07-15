#!/usr/bin/julia

include("multi-pairs-test-utils.jl")

const nsegs = [80, 100, 120, 140, 160, 180, 200]
const compute_bufs_opt = [SL.ComputeBuffer{nseg,Float64}(Val(SS.mask_allδ), Val(SS.mask_allδ)) for nseg in nsegs]
const preobjs = [get_preobj(modes, buf, 0.6) for buf in compute_bufs_opt]
const preopts = [PreOptimizer(preobj) for preobj in preobjs]

const solutions = Vector{NSegSolution}(undef, length(nsegs))
for i in 1:length(nsegs)
    precompile(run_preopts, (typeof(nsegs[i]), typeof(preopts[i]),
                             Int, Float64, Float64, Int))
end

@time Threads.@threads for i in 1:length(nsegs)
    solutions[i] = run_preopts(nsegs[i], preopts[i], 100, 150.0, 350.0, 25)
end
open("pairs_data.json", "w") do io
    println(io, JSON.json([todict(s1) for s1 in solutions]))
end
