#!/usr/bin/julia

push!(LOAD_PATH, joinpath(@__DIR__, "../../lib"))

using BenchmarkTools
using NLopt
using StaticArrays
using Statistics
using JSON

using MSSim
const Opts = MSSim.Optimizers
const SS = MSSim.SegSeq
const SL = MSSim.SymLinear
const Seq = MSSim.Sequence
const U = MSSim.Utils

struct PreOptObjective{NModes}
end

function (::PreOptObjective{NModes})(vals) where NModes
    dis = vals[1]
    disδ = vals[2]
    area = zero(eltype(vals))
    # areaδ = zero(eltype(vals))
    for i in 1:NModes
        # areai = abs(vals[2 + i])
        # areaδi = abs(vals[2 + NModes + i])
        area = max(area, abs(vals[2 + i]))
        # areaδ += areaδi / areai
    end
    return (dis + disδ / 200) / area^2
end

function get_preobj(modes, buf, ratio=0.6)
    nmodes = length(modes.modes)
    area_args = ntuple(i->(:area, i), nmodes)
    areaδ_args = ntuple(i->(:areaδ, i), nmodes)
    return Seq.Objective(SL.pmask_tfm,
                         ((:dis2, 0), (:disδ2, 0), area_args..., areaδ_args...),
                         Opts.autodiff(PreOptObjective{7}()), modes, buf,
                         freq=Seq.FreqSpec(true, sym=false),
                         amp=Seq.AmpSpec(cb=U.BlackmanStartEnd{ratio}()))
end

struct PreOptimizer{NModes,ObjArgs,Obj}
    preobj::Obj
    tracker::Opts.NLVarTracker
    opt::NLopt.Opt
end
function PreOptimizer(preobj::Obj;
                      τmin=0.1, τmax=3, Ω=0.4, ωmin=2π * 2.37, ωmax=2π * 2.52) where Obj
    nargs = Seq.nparams(preobj)
    tracker = Opts.NLVarTracker(nargs)
    Opts.set_bound!(tracker, preobj.param.τ, τmin, τmax)
    Opts.set_bound!(tracker, preobj.param.Ωs[1], Ω, Ω)
    for ω in preobj.param.ωs
        Opts.set_bound!(tracker, ω, ωmin, ωmax)
    end

    nmodes = length(preobj.modes)

    opt = NLopt.Opt(:LD_LBFGS, nargs)
    NLopt.min_objective!(opt, preobj)
    NLopt.maxeval!(opt, 2000)

    area_args = ntuple(i->(:area, i), nmodes)
    areaδ_args = ntuple(i->(:areaδ, i), nmodes)

    o = PreOptimizer{nmodes,((:τ, 0), (:dis2, 0), (:disδ2, 0), area_args..., areaδ_args...),Obj}(preobj, tracker, opt)
    update_bounds!(o)
    return o
end

function update_bounds!(o::PreOptimizer)
    NLopt.lower_bounds!(o.opt, Opts.lower_bounds(o.tracker))
    NLopt.upper_bounds!(o.opt, Opts.upper_bounds(o.tracker))
end

struct SolutionInfo{NModes}
    params::Vector{Float64}
    total_t::Float64
    dis2::Float64
    disδ2::Float64
    area::NTuple{NModes,Float64}
    areaδ::NTuple{NModes,Float64}
end

function todict(info::SolutionInfo)
    return Dict(
        "params"=>info.params,
        "total_t"=>info.total_t,
        "dis2"=>info.dis2,
        "disδ2"=>info.disδ2,
        "area"=>collect(info.area),
        "areaδ"=>collect(info.areaδ)
    )
end

function fromdict(::Type{SolutionInfo}, d)
    params = d["params"]
    total_t = d["total_t"]
    dis2 = d["dis2"]
    disδ2 = d["disδ2"]
    area = d["area"]
    areaδ = d["areaδ"]
    nmodes = length(area)
    @assert length(areaδ) == nmodes
    return SolutionInfo{nmodes}(params, total_t, dis2, disδ2, (area...,), (areaδ...,))
end

function compute_one(o::PreOptimizer{NModes,ObjArgs}) where {NModes,ObjArgs}
    obj, params, ret = NLopt.optimize(o.opt, Opts.init_vars!(o.tracker))
    if getfield(NLopt, ret) < 0
        return
    end
    info = o.preobj(Val(ObjArgs), params) do x
        return SolutionInfo{NModes}(params, x[1], x[2], x[3], ntuple(i->x[i + 3], NModes),
                                    ntuple(i->x[i + 3 + NModes], NModes))
    end
    area = 0.0
    areaδ = 0.0
    for i in 1:NModes
        areai = abs(info.area[i])
        areaδi = abs(info.areaδ[i])
        area += areai
        areaδ += areaδi / areai
    end
    if abs(info.dis2) < 1e-6 && abs(info.disδ2) < 1e-4 && area >= 15 * NModes
        return info
    end
end

function find_candidates!(o::PreOptimizer, nrounds, candidates=SolutionInfo[])
    for i in 1:nrounds
        info = compute_one(o)
        if info !== nothing
            push!(candidates, info)
        end
    end
    return candidates
end

struct TimeRangeSolution
    total_t_min::Float64
    total_t_max::Float64
    solutions::Vector{SolutionInfo}
end

function todict(sol::TimeRangeSolution)
    return Dict(
        "total_t_min"=>sol.total_t_min,
        "total_t_max"=>sol.total_t_max,
        "solutions"=>[todict(s) for s in sol.solutions]
    )
end

function fromdict(::Type{TimeRangeSolution}, d)
    return TimeRangeSolution(d["total_t_min"], d["total_t_max"],
                             [fromdict(SolutionInfo, s) for s in d["solutions"]])
end

function merge_into!(sol::TimeRangeSolution, other::TimeRangeSolution)
    @assert sol.total_t_min == other.total_t_min
    @assert sol.total_t_max == other.total_t_max
    append!(sol.solutions, other.solutions)
    return
end

struct NSegSolution
    nseg::Int
    solutions::Vector{TimeRangeSolution}
end

function todict(sol::NSegSolution)
    return Dict(
        "nseg"=>sol.nseg,
        "solutions"=>[todict(s) for s in sol.solutions]
    )
end

function fromdict(::Type{NSegSolution}, d)
    return NSegSolution(d["nseg"],
                        [fromdict(TimeRangeSolution, s) for s in d["solutions"]])
end

function merge_into!(sol::NSegSolution, other::NSegSolution)
    @assert sol.nseg == other.nseg
    time_pos = Dict{NTuple{2,Float64},Int}()
    for (i, s) in enumerate(sol.solutions)
        time_pos[(s.total_t_min, s.total_t_max)] = i
    end
    for s in other.solutions
        i = get!(time_pos, (s.total_t_min, s.total_t_max)) do
            push!(sol.solutions,
                  TimeRangeSolution(s.total_t_min, s.total_t_max, SolutionInfo[]))
            return length(sol.solutions)
        end
        merge_into!(sol.solutions[i], s)
    end
    return
end

function run_preopts(nseg, o::PreOptimizer, rounds, total_t_min, total_t_max, nsteps)
    res = TimeRangeSolution[]
    for i in 1:nsteps
        τmin = total_t_min + (total_t_max - total_t_min) / nsteps * (i - 1)
        τmax = total_t_min + (total_t_max - total_t_min) / nsteps * i
        println("Testing total time range [$(τmin), $(τmax)]")
        Opts.set_bound!(o.tracker, o.preobj.param.τ, τmin / nseg, τmax / nseg)
        update_bounds!(o)
        @time push!(res, TimeRangeSolution(τmin, τmax, find_candidates!(o, rounds)))
        println("Found $(length(res[end].solutions)) candidates")
    end
    return NSegSolution(nseg, res)
end

const radial_modes = [
    2.349490,
    2.397870,
    2.439980,
    2.476530,
    2.507130,
    2.531490,
    2.548960
]
const lamb_dicke_parameters = [
    0.12535185674701685,
    0.12411265750465082,
    0.12305413516480627,
    0.12216335005557813,
    0.12143343371677694,
    0.12086534864431214,
    0.12047167276221905
]
const participation_factors = [
    0.02222009251306859 -0.1722648656984975 0.48939776715759997 -0.678705987944334 0.4893977671575958 -0.17226486569849606 0.022220092513065576
    -0.08507556148734259 0.4120558633416861 -0.5683063560469468 0.0 0.5683063560469427 -0.4120558633416741 0.08507556148733478
    -0.21303936535968387 0.5714074376873661 -0.11990319870608773 -0.4769297472432097 -0.1199031987060997 0.5714074376873872 -0.2130393653596884
    0.39521513624944005 -0.44499802171470876 -0.38181377234109953 0.0 0.38181377234109876 0.4449980217146981 -0.39521513624943017
    0.5579098076201139 -0.031003440845807535 -0.3213345695440472 -0.41114359446049004 -0.3213345695440565 -0.031003440845817597 0.5579098076201152
    -0.5801440725517667 -0.3635749250921281 -0.1767657459105281 0.0 0.176765745910517 0.3635749250921191 0.5801440725517734
    0.37796447300922137 0.37796447300923375 0.37796447300922364 0.3779644730092235 0.37796447300922775 0.37796447300923464 0.37796447300922603
]

function mode_weight!(weights, ion1, ion2)
    for i in 1:7
        weights[i] = participation_factors[i, ion1] * participation_factors[i, ion2] * lamb_dicke_parameters[i]^2
    end
    return weights
end

const modes = Seq.Modes()
for i in 1:7
    push!(modes, 2π * radial_modes[i], 1)
end
