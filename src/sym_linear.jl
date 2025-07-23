#!/usr/bin/julia

module SymLinear

using StaticArrays

# Integral for pulses that are piecewise linear in both amplitude and phase.

module SegInt

import ...Utils as U
import ...SegSeq

using StaticArrays

# Generate a structure with the trig ratios we need precomputed
# It's easier for the compiler to do DCE on the values we don't use
# then to do CSE on branchy code that are duplicated.
function _trig_field_name(name)
    m = match(r"^sin_c([_0-9]*)$", name)
    if m !== nothing
        return Symbol("S_C$(m[1])")
    end
    m = match(r"^sin_f([_0-9]*)$", name)
    if m !== nothing
        return Symbol("S$(m[1])")
    end
    m = match(r"^cos_f([_0-9]*)$", name)
    if m !== nothing
        return Symbol("C$(m[1])")
    end
end

macro gen_trig_ratios(d, s, c, names...)
    expr = :(())
    for (name::Symbol) in names
        field_name = _trig_field_name(String(name))
        push!(expr.args, :($field_name = U.$name($(esc(d)), $(esc(s)), $(esc(c)))))
    end
    return expr
end

# Integral for each segments
# The kernel version are shared by both the test version
# and the version used in actual computation.
@inline function displacement_kernel(o, o′, d, s, c, V)
    return complex(muladd(o + o′, V.S_C1, -U.mul(o′, V.C1)),
                   muladd(U.mul(o, d), V.C1, U.mul(o′, V.S_C2)))
end

@inline function displacement_δ_kernel(o, o′, d, s, c, V)
    return complex(muladd(o + o′, -V.S_C2, U.mul(o′, V.C2)),
                   muladd(o, muladd(-d, V.C2, V.C1), U.mul(o′, V.S_C3)))
end

@inline function displacement_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    τ2 = τ^2
    return (U.mul(muladd(Ω′, τ, Ω), complex(c, s)),
            τ * complex(V.S_C1, d * V.C1),
            τ2 * complex(V.S_C1 - V.C1, V.S_C2))
end

@inline function displacement_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    τ2 = τ^2
    τ3 = τ2 * τ
    return (U.mul(o + o′, complex(-s, c)),
            τ2 * complex(-V.S_C2, muladd(d, -V.C2, V.C1)),
            τ3 * complex(V.C2 - V.S_C2, V.S_C3),
            τ2 * complex(muladd(o + o′, -V.S_C3, U.mul(o′, V.C3_2)),
                          -muladd(o′, V.S3_3, U.mul(o, muladd(d, V.C3_2, 2 * V.C2)))))
end

@inline function cumulative_displacement_kernel(o, o′, d, s, c, V)
    return complex(muladd(o, V.C1, U.mul(o′, V.S2)),
                   muladd(o, U.mul(V.S1, d), U.mul(o′, V.C2)))
end

@inline function cumulative_displacement_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    τ2 = τ^2
    τ3 = τ2 * τ
    return (complex(muladd(o + o′, V.S_C1, -U.mul(o′, V.C1)),
                    muladd(U.mul(o, d), V.C1, U.mul(o′, V.S_C2))),
            τ2 * complex(V.C1, U.mul(V.S1, d)),
            τ3 * complex(V.S2, V.C2),
            τ2 * complex(-muladd(o, V.C2, U.mul(o′, V.S3_2)),
                          muladd(o, V.S2, U.mul(o′, V.C3_2))))
end

# Twice the enclosed area
@inline function enclosed_area_complex_kernel(o, o′, d, s, c, V)
    a1 = o * (o + o′)
    a2 = o′^2
    return complex(muladd(a1, V.C1, U.mul(a2, V.C3)),
                   muladd(a1, U.mul(V.S1, d), U.mul(a2, V.S3)))
end

# Twice the enclosed area
@inline function enclosed_area_kernel(o, o′, d, s, c, V)
    a1 = o * (o + o′)
    a2 = o′^2
    return muladd(a1, U.mul(V.S1, d), U.mul(a2, V.S3))
end

@inline function enclosed_area_δ_kernel(o, o′, d, s, c, V)
    a1 = o * (o + o′)
    a2 = o′^2
    return muladd(a1, V.S2, U.mul(a2, V.S4))
end

@inline function enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    τ2 = τ^2
    return (muladd(Ω′, τ, Ω) * d * muladd(o, V.C1, U.mul(o′, V.S1)),
            τ * muladd(2, o, o′) * (V.S1 * d),
            τ2 * muladd(U.mul(2, o′), V.S3, U.mul(o, U.mul(V.S1, d))))
end

@inline function enclosed_area_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    τ2 = τ^2
    τ3 = τ2 * τ
    return (muladd(muladd(muladd(-2, V.S1, V.S_C1), o′, U.mul(o, V.S_C1 - V.C1)),
                   o, U.mul(U.mul(o′, o′), V.S2)),
            τ2 * muladd(2, o, o′) * V.S2,
            τ3 * muladd(U.mul(2, o′), V.S4, U.mul(o, V.S2)),
            τ2 * muladd(U.mul(o, o + o′), -V.S3_2, -U.mul(U.mul(o′, o′), V.S5)))
end

# These are for testing only.
# The `compute_values` below is the one that's used in actual computation.
function displacement(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_c1, sin_c2, cos_f1)
    return phase0 * displacement_kernel(o, o′, d, s, c, V)
end

function displacement_δ(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, cos_f2, sin_c2, sin_c3)
    return phase0 * τ * displacement_δ_kernel(o, o′, d, s, c, V)
end

function displacement_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_c1, sin_c2, sin_c3, cos_f1, cos_f2)

    τΩs = displacement_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (phase0 * τΩs[1], phase0 * τΩs[2], phase0 * τΩs[3],
            U.mulim(phase0 * displacement_kernel(o, o′, d, s, c, V)),
            phase0 * τ * displacement_δ_kernel(o, o′, d, s, c, V))
end

function displacement_δ_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, cos_f2, cos_f3_2, sin_f3_3, sin_c2, sin_c3)

    τΩsδ = displacement_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (phase0 * τΩsδ[1], phase0 * τΩsδ[2], phase0 * τΩsδ[3],
            U.mulim(phase0 * τ * displacement_δ_kernel(o, o′, d, s, c, V)),
            phase0 * τΩsδ[4])
end

function cumulative_displacement(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, cos_f2, sin_f2)

    return phase0 * τ * cumulative_displacement_kernel(o, o′, d, s, c, V)
end

function cumulative_displacement_gradients(τ, Ω, Ω′, φ, δ)
    phase0 = cis(φ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, cos_f2, sin_f2,
                         sin_f3_2, cos_f3_2, sin_c1, sin_c2)

    τΩsδ = cumulative_displacement_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (phase0 * τΩsδ[1], phase0 * τΩsδ[2], phase0 * τΩsδ[3],
            U.mulim(phase0 * τ * cumulative_displacement_kernel(o, o′, d, s, c, V)),
            phase0 * τΩsδ[4])
end

# Twice the enclosed area
function enclosed_area_complex(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, cos_f3, sin_f3)

    return enclosed_area_complex_kernel(o, o′, d, s, c, V)
end

# Twice the enclosed area
function enclosed_area(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_f1, sin_f3)

    return enclosed_area_kernel(o, o′, d, s, c, V)
end

function enclosed_area_δ(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, sin_f2, sin_f4)

    return τ * enclosed_area_δ_kernel(o, o′, d, s, c, V)
end

function enclosed_area_gradients(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, sin_f2, sin_f3, sin_f4)

    τΩs = enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (τΩs[1], τΩs[2], τΩs[3], zero(φ),
            τ * enclosed_area_δ_kernel(o, o′, d, s, c, V))
end

function enclosed_area_δ_gradients(τ, Ω, Ω′, φ, δ)
    d = δ * τ
    o = Ω * τ
    o′ = Ω′ * τ^2
    s, c = U.fast_sincos(d)
    V = @gen_trig_ratios(d, s, c, cos_f1, sin_f1, sin_f2, sin_f3_2,
                         sin_f4, sin_f5, sin_c1)

    τΩsδ = enclosed_area_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
    return (τΩsδ[1], τΩsδ[2], τΩsδ[3], zero(φ), τΩsδ[4])
end

# The values we may care about in each segments
# * Displacement (dis)
# * Gradient of displacement w.r.t. detuning (disδ)
# * Cumulative displacement (cumdis)
# * Enclosed area (area)
# * Gradient of enclosed area w.r.t. detuning (areaδ)
# As well as the gradient of everything above w.r.t. each of the input parameters

# Compute all the values we want for this segment in one go.
# This should allow the compiler to reuse many of the intermediate results
# when computing different values.
@inline function (compute_values(τ::_T, Ω, Ω′, φ, δ, ::Val{maskv}, ::Val{maskg})
                  where {_T,maskv,maskg})

    T = float(_T)
    CT = Complex{T}
    SDV = SegSeq.SegData(T, maskv)
    SDG = SegSeq.SegData(T, maskg)

    need_grad = maskg !== zero(SegSeq.ValueMask)

    @inline begin
        d = δ * τ
        o = Ω * τ
        o′ = Ω′ * τ^2
        s, c = U.fast_sincos(d)
        sφ, cφ = U.fast_sincos(φ)
        phase0 = complex(cφ, sφ)
        phase0_τ = phase0 * τ
        V = @gen_trig_ratios(d, s, c, sin_c1, sin_c2, sin_c3,
                             cos_f1, sin_f1, cos_f2, sin_f2,
                             cos_f3_2, sin_f3, sin_f3_2, sin_f3_3, sin_f4, sin_f5)

        dis = U.mul(phase0, displacement_kernel(o, o′, d, s, c, V))
        area = enclosed_area_kernel(o, o′, d, s, c, V)
        cumdis = U.mul(phase0_τ, cumulative_displacement_kernel(o, o′, d, s, c, V))
        disδ = U.mul(phase0_τ, displacement_δ_kernel(o, o′, d, s, c, V))
        areaδ = τ * enclosed_area_δ_kernel(o, o′, d, s, c, V)
        res = SDV(maskv.τ ? τ : nothing, maskv.dis ? dis : nothing,
                  maskv.area ? area : nothing, maskv.cumdis ? cumdis : nothing,
                  maskv.disδ ? disδ : nothing, maskv.areaδ ? areaδ : nothing)
        if !need_grad
            return res, nothing
        end
        if maskg.τ
            τ_grad = SA[one(T), zero(T), zero(T), zero(T), zero(T)]
        else
            τ_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        if maskg.dis
            dis_τΩs = displacement_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
            dis_grad = SA[U.mul(phase0, dis_τΩs[1]), U.mul(phase0, dis_τΩs[2]),
                          U.mul(phase0, dis_τΩs[3]), U.mulim(dis), disδ]
        else
            dis_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        if maskg.area
            area_τΩs = enclosed_area_τΩs_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
            area_grad = SA[area_τΩs[1], area_τΩs[2], area_τΩs[3], zero(T), areaδ]
        else
            area_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        if maskg.cumdis
            cumdis_τΩsδ = cumulative_displacement_τΩsδ_kernel(o, o′, d, s, c,
                                                                    Ω, Ω′, τ, V)
            cumdis_grad = SA[U.mul(phase0, cumdis_τΩsδ[1]),
                             U.mul(phase0, cumdis_τΩsδ[2]),
                             U.mul(phase0, cumdis_τΩsδ[3]),
                             U.mulim(cumdis),
                             U.mul(phase0, cumdis_τΩsδ[4])]
        else
            cumdis_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        if maskg.disδ
            disδ_τΩsδ = displacement_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
            disδ_grad = SA[U.mul(phase0, disδ_τΩsδ[1]),
                            U.mul(phase0, disδ_τΩsδ[2]),
                            U.mul(phase0, disδ_τΩsδ[3]),
                            U.mulim(disδ),
                            U.mul(phase0, disδ_τΩsδ[4])]
        else
            disδ_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        if maskg.areaδ
            areaδ_τΩsδ = enclosed_area_δ_τΩsδ_kernel(o, o′, d, s, c, Ω, Ω′, τ, V)
            areaδ_grad = SA[areaδ_τΩsδ[1], areaδ_τΩsδ[2], areaδ_τΩsδ[3],
                             zero(T), areaδ_τΩsδ[4]]
        else
            areaδ_grad = SA[nothing, nothing, nothing, nothing, nothing]
        end
        grads = SA[SDG(τ_grad[1], dis_grad[1], area_grad[1], cumdis_grad[1],
                       disδ_grad[1], areaδ_grad[1]),
                   SDG(τ_grad[2], dis_grad[2], area_grad[2], cumdis_grad[2],
                       disδ_grad[2], areaδ_grad[2]),
                   SDG(τ_grad[3], dis_grad[3], area_grad[3], cumdis_grad[3],
                       disδ_grad[3], areaδ_grad[3]),
                   SDG(τ_grad[4], dis_grad[4], area_grad[4], cumdis_grad[4],
                       disδ_grad[4], areaδ_grad[4]),
                   SDG(τ_grad[5], dis_grad[5], area_grad[5], cumdis_grad[5],
                       disδ_grad[5], areaδ_grad[5])]
        return res, grads
    end
end

end

import ..Utils as U
import ..SegSeq

struct ParamGradMask
    τ::Bool
    Ω::Bool
    Ω′::Bool
    φ::Bool
    ω::Bool
end
Base.zero(::Type{ParamGradMask}) = ParamGradMask(false, false, false, false, false)
const pmask_full = ParamGradMask(true, true, true, true, true)
const pmask_fm = ParamGradMask(false, false, false, true, true)
const pmask_tfm = ParamGradMask(true, false, false, true, true)
const pmask_am = ParamGradMask(false, true, true, false, false)
const pmask_tam = ParamGradMask(true, true, true, false, false)

struct ComputeBuffer{NSeg,T,SDV<:SegSeq.SegData{T},SDG<:SegSeq.SegData{T}}
    seg_buf::Vector{SDV}
    seg_grad_buf::U.JaggedMatrix{SDG}
    buffer::SegSeq.SeqComputeBuffer{T}

    function ComputeBuffer{NSeg,T}(::Val{maskv}, ::Val{maskg}) where {NSeg,T,maskv,maskg}
        SDV = SegSeq.SegData(T, maskv)
        SDG = SegSeq.SegData(T, maskg)

        seg_buf = Vector{SDV}(undef, NSeg)
        seg_grad_buf = U.JaggedMatrix{SDG}()
        buffer = SegSeq.SeqComputeBuffer{T}()
        U.resize_uniform!(seg_grad_buf, NSeg, 5)
        return new{NSeg,T,SDV,SDG}(seg_buf, seg_grad_buf, buffer)
    end
end

mutable struct Kernel{NSeg,T,SDV<:SegSeq.SegData{T},SDG<:SegSeq.SegData{T},pmask,NArgs}
    const buffer::ComputeBuffer{NSeg,T,SDV,SDG}
    const result::SegSeq.SingleModeResult{T,SDV,SDG}
    evaled::Bool
    const args::MVector{NArgs,T}

    function Kernel(buffer::ComputeBuffer{NSeg,T,SDV,SDG},
                    ::Val{pmask}) where {NSeg,T,SDV,SDG,pmask}
        maskv = SegSeq.value_mask(SDV)
        maskg = SegSeq.value_mask(SDG)
        result = SegSeq.SingleModeResult{T}(Val(maskv), Val(maskg))
        U.resize_uniform!(result.grad, NSeg, 5)
        return new{NSeg,T,SDV,SDG,pmask,NSeg*5}(buffer, result, false,
                                                MVector{NSeg*5,T}(undef))
    end
end

# Arguments
# (τ, Ω, Ω′, φ, δ) * nseg
function force_update!(kern::Kernel{NSeg,T,SDV,SDG,pmask,NArgs}) where {NSeg,T,SDV,SDG,pmask,NArgs}
    buffer = kern.buffer
    maskv = SegSeq.value_mask(SDV)
    maskg = SegSeq.value_mask(SDG)
    seg_buf = buffer.seg_buf
    seg_grad_buf = buffer.seg_grad_buf
    need_grad = maskg !== zero(SegSeq.ValueMask)
    args = kern.args
    @inbounds for i in 1:NSeg
        τ = args[i * 5 - 4]
        Ω = args[i * 5 - 3]
        Ω′ = args[i * 5 - 2]
        φ = args[i * 5 - 1]
        δ = args[i * 5]
        if Ω′ == 0
            seg, grad = SegInt.compute_values(τ, Ω, U.Zero(), φ, δ,
                                              Val(maskv), Val(maskg))
        else
            seg, grad = SegInt.compute_values(τ, Ω, Ω′, φ, δ, Val(maskv), Val(maskg))
        end
        seg_buf[i] = seg
        if need_grad
            grad_buf = seg_grad_buf[i]
            grad_buf[1] = pmask.τ ? grad[1] : SDG()
            grad_buf[2] = pmask.Ω ? grad[2] : SDG()
            grad_buf[3] = pmask.Ω′ ? grad[3] : SDG()
            grad_buf[4] = pmask.φ ? grad[4] : SDG()
            grad_buf[5] = pmask.ω ? grad[5] : SDG()
        end
    end
    SegSeq.compute_single_mode!(kern.result, seg_buf, buffer.buffer,
                                need_grad ? seg_grad_buf : nothing,
                                Val(pmask.τ), Val(true))
    return
end

@inline function eval_with_mode!(kern::Kernel{NSeg}, args, ωm) where NSeg
    φ = 0.0
    kargs = kern.args
    @inbounds for j in 1:NSeg
        τ = args[j * 5 - 4]
        kargs[j * 5 - 4] = τ
        kargs[j * 5 - 3] = args[j * 5 - 3]
        kargs[j * 5 - 2] = args[j * 5 - 2]
        kargs[j * 5 - 1] = args[j * 5 - 1] - φ
        δ = args[j * 5]
        kargs[j * 5] = δ - ωm
        φ = muladd(ωm, τ, φ)
    end
    force_update!(kern)
end

@inline function update!(kern::Kernel{NSeg,T,SDV,SDG,pmask,NArgs},
                         args::NTuple{N}) where {NSeg,T,SDV,SDG,pmask,NArgs,N}
    @assert N == NArgs
    same = true
    kargs = kern.args
    @inbounds @simd ivdep for i in 1:NArgs
        arg = args[i]
        same &= arg == kargs[i]
        kargs[i] = arg
    end
    if same && kern.evaled
        return
    end
    kern.evaled = false
    force_update!(kern)
    kern.evaled = true
    return
end

@inline _val_mask(kern::Kernel{NSeg,T,SDV,SDG}) where {NSeg,T,SDV,SDG} = SegSeq.value_mask(SDV)
@inline _grad_mask(kern::Kernel{NSeg,T,SDV,SDG}) where {NSeg,T,SDV,SDG} = SegSeq.value_mask(SDG)
@inline _nseg(kern::Kernel{NSeg}) where NSeg = NSeg

for var in [:dis, :cumdis, :disδ]
    rvar = "r$var"
    ivar = "i$var"
    var2 = "$(var)2"
    @eval begin
        @inline function $(Symbol("value_$rvar"))(kern::Kernel, args...)
            @assert _val_mask(kern).$var
            update!(kern, args)
            return real(kern.result.val.$var)
        end
        @inline function $(Symbol("grad_$rvar"))(g, kern::Kernel, args...)
            @assert _grad_mask(kern).$var
            update!(kern, args)
            grad = kern.result.grad.values
            @inbounds for i in 1:_nseg(kern) * 5
                g[i] = real(grad[i].$var)
            end
        end
        @inline function $(Symbol("value_$ivar"))(kern::Kernel, args...)
            @assert _val_mask(kern).$var
            update!(kern, args)
            return imag(kern.result.val.$var)
        end
        @inline function $(Symbol("grad_$ivar"))(g, kern::Kernel, args...)
            @assert _grad_mask(kern).$var
            update!(kern, args)
            grad = kern.result.grad.values
            @inbounds for i in 1:_nseg(kern) * 5
                g[i] = imag(grad[i].$var)
            end
        end
        @inline function $(Symbol("value_$var2"))(kern::Kernel, args...)
            @assert _val_mask(kern).$var
            update!(kern, args)
            return abs2(kern.result.val.$var)
        end
        @inline function $(Symbol("grad_$var2"))(g, kern::Kernel, args...)
            @assert _grad_mask(kern).$var
            update!(kern, args)
            grad = kern.result.grad.values
            v2 = 2 * kern.result.val.$var
            @inbounds for i in 1:_nseg(kern) * 5
                gv = grad[i].$var
                g[i] = muladd(real(v2), real(gv), imag(v2) * imag(gv))
            end
        end
    end
end

for var in [:area, :areaδ]
    @eval begin
        @inline function $(Symbol("value_$var"))(kern::Kernel, args...)
            @assert _val_mask(kern).$var
            update!(kern, args)
            return kern.result.val.$var
        end
        @inline function $(Symbol("grad_$var"))(g, kern::Kernel, args...)
            @assert _grad_mask(kern).$var
            update!(kern, args)
            grad = kern.result.grad.values
            @inbounds for i in 1:_nseg(kern) * 5
                g[i] = grad[i].$var
            end
        end
    end
end

end
