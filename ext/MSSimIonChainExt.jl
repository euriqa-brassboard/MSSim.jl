module MSSimIonChainExt

import MSSim.IonChain: IonChain, IonInfo, Function1D,
    AxialPosInfo, AxialModel, _new_axial_model, set_init_pos!, update_all_init_pos!

using JuMP
using NLopt

function _register(model, potential::Function1D, name)
    if potential.∇²f !== nothing
        add_nonlinear_operator(model, 1, potential.f, potential.∇f, potential.∇²f,
                               name=name)
    elseif potential.∇f !== nothing
        add_nonlinear_operator(model, 1, potential.f, potential.∇f, name=name)
    else
        add_nonlinear_operator(model, 1, potential.f, name=name)
    end
end

function AxialModel(ions::Vector{IonInfo}, dc::Function1D,
                    rf::Union{Function1D,Nothing}=nothing; model=nothing)
    if model === nothing
        model = Model(NLopt.Optimizer)
        # LD_MMA is slower
        # LD_SLSQP is slightly faster but gives wrong answer from time to time
        # Other LD_ solvers does not accept the inequality constraints.
        set_optimizer_attribute(model, "algorithm", :LD_CCSAQ)
    end
    nions = length(ions)
    vars = [@variable(model) for i in 1:nions]
    for i in 2:nions
        @constraint(model, vars[i - 1] <= vars[i])
    end
    pos = [AxialPosInfo(NaN, -Inf, Inf) for i in 1:nions]
    dc = _register(model, dc, :dc)
    if rf !== nothing
        rf = _register(model, rf, :rf)
    end
    obj = 0
    for (i1, ion1) in enumerate(ions)
        pos1 = vars[i1]
        if rf !== nothing
            obj = @expression(model, obj + dc(pos1) * ion1.charge
                              + rf(pos1) * (ion1.charge / ion1.mass)^2)
        else
            obj = @expression(model, obj + dc(pos1) * ion1.charge)
        end
        for i2 in (i1 + 1):nions
            ion2 = ions[i2]
            pos2 = vars[i2]
            obj = @expression(model, obj + ion1.charge * ion2.charge / (pos2 - pos1))
        end
    end
    @objective(model, Min, obj)
    return _new_axial_model(model, ions, pos, vars)
end

function IonChain.optimize!(am::AxialModel,
                            pos_out::Vector=Vector{Float64}(undef, length(am.posvars::Vector{VariableRef})))
    posvars = am.posvars::Vector{VariableRef}
    for (info, var) in zip(am.pos, posvars)
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
    JuMP.optimize!(am.model::Model)
    pos_out .= value.(posvars)
    return pos_out
end

function update_all_init_pos!(am::AxialModel)
    for (i, var) in enumerate(am.posvars::Vector{VariableRef})
        set_init_pos!(am, i, value(var))
    end
    return am
end

end
