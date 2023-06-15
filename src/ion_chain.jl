#!/usr/bin/julia

module IonChain

using JuMP
using NLopt
using Setfield
using StaticArrays

struct IonInfo
    charge::Float64
    mass::Float64
end

simple_ions(n) = fill(IonInfo(1, 1), n)

struct Function1D{F,∇F,∇²F}
    f::F
    ∇f::∇F
    ∇²f::∇²F
    Function1D(f::F, ∇f::∇F=nothing, ∇²f::∇²F=nothing) where {F,∇F,∇²F} =
        new{F,∇F,∇²F}(f, ∇f, ∇²f)
end

function _register(model, potential::Function1D, name)
    if potential.∇²f !== nothing
        register(model, name, 1, potential.f, potential.∇f, potential.∇²f)
    elseif potential.∇f !== nothing
        register(model, name, 1, potential.f, potential.∇f)
    else
        register(model, name, 1, potential.f, autodiff=true)
    end
    return
end

function poly_function(::Val{N}) where N
    coeffs = zeros(MVector{N})
    function f(x)
        return x * evalpoly(x, Tuple(coeffs))
    end
    function ∇f(x)
        cs = Tuple(coeffs)
        cs2 = ntuple(i->i * cs[i], Val(N))
        return evalpoly(x, cs2)
    end
    function ∇²f(x)
        cs = Tuple(coeffs)
        cs2 = ntuple(i->i * (i + 1) * cs[i + 1], Val(N - 1))
        return evalpoly(x, cs2)
    end
    return coeffs, Function1D(f, ∇f, ∇²f)
end

struct AxialPosInfo
    pos::Float64
    pre_barrier::Float64
    post_barrier::Float64
end

struct AxialModel
    model::Model
    ions::Vector{IonInfo}
    pos::Vector{AxialPosInfo}
    posvars::Vector{VariableRef}
    function AxialModel(ions::Vector{IonInfo}, dc::Function1D,
                        rf::Union{Function1D,Nothing}=nothing; model=nothing)
        if model === nothing
            model = Model(NLopt.Optimizer)
            set_optimizer_attribute(model, "algorithm", :LD_SLSQP)
        end
        nions = length(ions)
        vars = [@variable(model) for i in 1:nions]
        for i in 2:nions
            @constraint(model, vars[i - 1] <= vars[i])
        end
        pos = [AxialPosInfo(NaN, -Inf, Inf) for i in 1:nions]
        _register(model, dc, :dc)
        if rf !== nothing
            _register(model, rf, :rf)
        end
        obj = @NLexpression(model, 0)
        for (i1, ion1) in enumerate(ions)
            pos1 = vars[i1]
            if rf !== nothing
                obj = @NLexpression(model, obj + dc(pos1) * ion1.charge
                                    + rf(pos1) * (ion1.charge / ion1.mass)^2)
            else
                obj = @NLexpression(model, obj + dc(pos1) * ion1.charge)
            end
            for i2 in (i1 + 1):nions
                ion2 = ions[i2]
                pos2 = vars[i2]
                obj = @NLexpression(model,
                                    obj + ion1.charge * ion2.charge / (pos2 - pos1))
            end
        end
        @NLobjective(model, Min, obj)
        return new(model, ions, pos, vars)
    end
end

function set_init_pos!(am::AxialModel, i, pos)
    if pos === nothing
        pos = NaN
    end
    pi = am.pos[i]
    @inbounds am.pos[i] = @set pi.pos = pos
    return am
end

function set_pre_barrier!(am::AxialModel, i, pos)
    if pos === nothing
        pos = -Inf
    end
    pi = am.pos[i]
    @inbounds am.pos[i] = @set pi.pre_barrier = pos
    return am
end

function set_post_barrier!(am::AxialModel, i, pos)
    if pos === nothing
        pos = -Inf
    end
    pi = am.pos[i]
    @inbounds am.pos[i] = @set pi.post_barrier = pos
    return am
end

# insert a barrier after the index i
function set_barrier!(am::AxialModel, i, pos)
    if i >= 1
        set_post_barrier(am, i, pos)
    end
    if i < length(am.pos)
        set_pre_barrier(am, i + 1, pos)
    end
    return am
end

function clear_barriers!(am::AxialModel)
    @inbounds for i in 1:length(am.pos)
        am.pos[i] = AxialPosInfo(am.pos[i].pos, -Inf, Inf)
    end
    return am
end

function optimize!(am::AxialModel,
                   pos_out::Vector=Vector{Float64}(undef, length(am.posvars)))
    for (info, var) in zip(am.pos, am.posvars)
        if isfinite(info.pos)
            set_start_value(var, info.pos)
        else
            set_start_value(var, nothing)
        end
        if isfinite(info.pre_barrier)
            set_lower_bound(var, info.pre_barrier)
        elseif has_lower_bound(var)
            delete_lower_bound(var)
        end
        if isfinite(info.post_barrier)
            set_upper_bound(var, info.post_barrier)
        elseif has_upper_bound(var)
            delete_upper_bound(var)
        end
    end
    JuMP.optimize!(am.model)
    pos_out .= value.(am.posvars)
    return pos_out
end

function update_all_init_pos!(am::AxialModel)
    for (i, var) in enumerate(am.posvars)
        set_init_pos!(am, i, value(var))
    end
    return am
end

end
