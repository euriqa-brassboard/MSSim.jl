#!/usr/bin/julia

module IonChain

using ForwardDiff
using LinearAlgebra
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

function _derivative_2nd(f::Function1D, x)
    if f.∇²f !== nothing
        return f.∇²f(x)
    elseif f.∇f !== nothing
        return ForwardDiff.derivative(f.∇f, x)
    else
        ∇f = x->ForwardDiff.derivative(f.f, x)
        return ForwardDiff.derivative(∇f, x)
    end
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
    model
    ions::Vector{IonInfo}
    pos::Vector{AxialPosInfo}
    posvars::Vector
    global _new_axial_model(model, ions, pos, vars) = new(model, ions, pos, vars)
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

function optimize! end

function update_all_init_pos! end

function axial_modes(ions, poses, dc::Function1D, rf::Union{Function1D,Nothing}=nothing)
    nions = length(ions)
    H = zeros(nions, nions)
    for i in 1:nions
        pos = poses[i]
        ion = ions[i]
        ∇²f = _derivative_2nd(dc, pos) * ion.charge
        if rf !== nothing
            ∇²f += _derivative_2nd(rf, pos) * (ion.charge / ion.mass)^2
        end
        H[i, i] = ∇²f / ion.mass
    end
    for i2 in 2:nions
        pos2 = poses[i2]
        ion2 = ions[i2]
        for i1 in 1:i2 - 1
            pos1 = poses[i1]
            ion1 = ions[i1]
            mass12 = sqrt(ion1.mass * ion2.mass)
            term = 2 / (pos2 - pos1)^3 * ion1.charge * ion2.charge
            H[i1, i1] += term / ion1.mass
            H[i2, i2] += term / ion2.mass
            H[i1, i2] -= term / mass12
            H[i2, i1] -= term / mass12
        end
    end
    evs, vecs = eigen(H)
    return [ev >= 0 ? sqrt(ev) : -sqrt(-ev) for ev in evs], vecs
end

# Input function computes the second order derivative of the RF potential
# This assumes that the good axis for radial motion does not depend on
# the ion position, otherwise we need to solve all radial modes at the same time...
function radial_modes(ions, poses, dc::Union{Function1D,Nothing},
                      rf::Union{Function1D,Nothing}=nothing)
    nions = length(ions)
    H = zeros(nions, nions)
    for i in 1:nions
        pos = poses[i]
        ion = ions[i]
        ∇²f = 0.0
        if dc !== nothing
            ∇²f += dc.f(pos) * ion.charge
        end
        if rf !== nothing
            ∇²f += rf.f(pos) * (ion.charge / ion.mass)^2
        end
        H[i, i] = ∇²f / ion.mass
    end
    for i2 in 2:nions
        pos2 = poses[i2]
        ion2 = ions[i2]
        for i1 in 1:i2 - 1
            pos1 = poses[i1]
            ion1 = ions[i1]
            mass12 = sqrt(ion1.mass * ion2.mass)
            term = 1 / (pos2 - pos1)^3 * ion1.charge * ion2.charge
            H[i1, i1] -= term / ion1.mass
            H[i2, i2] -= term / ion2.mass
            H[i1, i2] += term / mass12
            H[i2, i1] += term / mass12
        end
    end
    evs, vecs = eigen(H)
    return [ev >= 0 ? sqrt(ev) : -sqrt(-ev) for ev in evs], vecs
end

end
