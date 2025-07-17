#!/usr/bin/julia

module Optimizers

import ForwardDiff

struct NLVarTracker
    vars::Vector{Tuple{Float64,Float64}}
    NLVarTracker(nargs) = new(fill((-Inf, Inf), nargs))
end

function set_bound!(tracker::NLVarTracker, idx, lb, ub)
    tracker.vars[idx] = (lb, ub)
    return
end

lower_bounds(tracker::NLVarTracker) = [lb for (lb, ub) in tracker.vars]
upper_bounds(tracker::NLVarTracker) = [ub for (lb, ub) in tracker.vars]

function init_vars!(tracker::NLVarTracker, vars=nothing)
    nvars = length(tracker.vars)
    if vars === nothing
        vars = Vector{Float64}(undef, nvars)
    end
    for vi in 1:nvars
        lb, ub = tracker.vars[vi]
        if isfinite(lb)
            if isfinite(ub)
                vars[vi] = lb + (ub - lb) * rand()
            else
                vars[vi] = lb + rand()
            end
        elseif isfinite(ub)
            vars[vi] = ub - rand()
        else
            vars[vi] = rand()
        end
    end
    return vars
end

function autodiff(f::F) where F
    function fn_with_diff(x, grad)
        if !isempty(grad)
            # Use ForwardDiff to compute the gradient. Replace with your
            # favorite Julia automatic differentiation package.
            ForwardDiff.gradient!(grad, f, x)
        end
        return f(x)
    end
end

end
