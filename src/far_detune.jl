#

module FarDetune

using Static
using StaticArrays

macro accum_grad(keep, tgt, g, w)
    tgt = esc(tgt)
    g = esc(g)
    w = esc(w)
    # Use an dummy expression to make sure the original source info is preserved
    :(nothing;
      if dynamic($(esc(keep)))
          $tgt = muladd($g, $w, $tgt)
      else
          $tgt = $g * $w
      end)
end

struct SlotArray{T}
    slots::T
    @inline SlotArray(slots...) = new{typeof(slots)}(slots)
end
@inline Base.isempty(a::SlotArray) = isempty(a.slots)
# @inline Base.length(a::SlotArray) = length(a.slots)
Base.@propagate_inbounds Base.getindex(a::SlotArray, i) = a.slots[i][]
Base.@propagate_inbounds Base.setindex!(a::SlotArray, v, i) = (a.slots[i][] = v)

@inline function enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ, grad,
                                      weight=static(true);
                                      keep_grad=(τ=static(false), δ=static(false),
                                                 Ω11=static(false), Ω12=static(false),
                                                 Ω21=static(false), Ω22=static(false)))
    δ⁻¹ = 1 / δ
    δ⁻¹_12 = δ⁻¹ / 12
    A = muladd(2, Ω21, Ω22)
    B = muladd(2, Ω22, Ω21)
    grad_τ = (A * Ω11 + B * Ω12) * δ⁻¹_12

    res = τ * grad_τ * weight

    @inbounds if !isempty(grad)
        w2 = τ * δ⁻¹_12 * weight
        @accum_grad(keep_grad.τ, grad[1], grad_τ, weight)
        @accum_grad(keep_grad.Ω11, grad[2], A, w2)
        @accum_grad(keep_grad.Ω12, grad[3], B, w2)
        @accum_grad(keep_grad.Ω21, grad[4], muladd(2, Ω11, Ω12), w2)
        @accum_grad(keep_grad.Ω22, grad[5], muladd(2, Ω12, Ω11), w2)
        @accum_grad(keep_grad.δ, grad[6], -res, δ⁻¹)
    end
    return res
end

@inline function enclosed_area_modes(τ, Ω11, Ω12, Ω21, Ω22, δ, ωs, weights, grad;
                                     keep_grad=(τ=static(false), δ=static(false),
                                                Ω11=static(false), Ω12=static(false),
                                                Ω21=static(false), Ω22=static(false)))
    keep_all_grad = (τ=static(true), δ=static(true), Ω11=static(true), Ω12=static(true),
                     Ω21=static(true), Ω22=static(true))
    nmodes = length(ωs)
    if nmodes == 0
        return enclosed_area_kernel(zero(τ), zero(Ω11), zero(Ω12), zero(Ω21), zero(Ω22),
                                    one(δ) - zero(eltype(ωs)),
                                    grad, zero(eltype(weights)); keep_grad=keep_grad)
    end
    res = @inbounds enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ - ωs[1], grad,
                                         weights[1]; keep_grad=keep_grad)
    @inbounds for i in 2:nmodes
        res += enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ - ωs[i], grad,
                                    weights[i]; keep_grad=keep_all_grad)
    end
    return res
end

@inline function enclosed_area_seq(τs, Ω1s, Ω2s, δs, ωs, weights,
                                   τs_g, Ω1s_g, Ω2s_g, δs_g;
                                   keep_grad=(τ=static(false), δ=static(false),
                                              Ω1s=static(false), Ω2s=static(false)))
    nsegs = length(τs)
    keep_grad0 = (τ=keep_grad.τ, δ=keep_grad.δ, Ω11=keep_grad.Ω1s, Ω12=keep_grad.Ω1s,
                  Ω21=keep_grad.Ω2s, Ω22=keep_grad.Ω2s)
    keep_grad1 = (τ=keep_grad.τ, δ=keep_grad.δ, Ω11=static(true), Ω12=keep_grad.Ω1s,
                  Ω21=static(true), Ω22=keep_grad.Ω2s)
    @inbounds if isempty(τs_g)
        res = enclosed_area_modes(τs[1], Ω1s[1], Ω1s[2], Ω2s[1], Ω2s[2], δs[1],
                                  ωs, weights, ())
        for i in 2:nsegs
            res += enclosed_area_modes(τs[i], Ω1s[i], Ω1s[i + 1], Ω2s[i], Ω2s[i + 1],
                                       δs[i], ωs, weights, ())
        end
    else
        res = enclosed_area_modes(τs[1], Ω1s[1], Ω1s[2], Ω2s[1], Ω2s[2], δs[1],
                                  ωs, weights,
                                  SlotArray(@view(τs_g[1]), @view(Ω1s_g[1]),
                                            @view(Ω1s_g[2]), @view(Ω2s_g[1]),
                                            @view(Ω2s_g[2]), @view(δs_g[1]));
                                  keep_grad=keep_grad0)
        for i in 2:nsegs
            res += enclosed_area_modes(τs[i], Ω1s[i], Ω1s[i + 1], Ω2s[i], Ω2s[i + 1],
                                       δs[i], ωs, weights,
                                       SlotArray(@view(τs_g[i]), @view(Ω1s_g[i]),
                                                 @view(Ω1s_g[i + 1]), @view(Ω2s_g[i]),
                                                 @view(Ω2s_g[i + 1]), @view(δs_g[i]));
                                       keep_grad=keep_grad1)
        end
    end
    return res
end

struct AreaTargets{NIons,Vec}
    targets::Vec
    weights::Vec
    function AreaTargets{NIons}() where NIons
        NPairs = NIons * (NIons - 1) ÷ 2
        targets = MVector{NPairs,Float64}(undef)
        weights = MVector{NPairs,Float64}(undef)
        # broadcast and fill! flattens the loop which is not necessary here.
        @inbounds for i in 1:NPairs
            targets[i] = 0
            weights[i] = 1
        end
        return new{NIons,typeof(targets)}(targets, weights)
    end
end

@noinline throw_boundserror(A, I) = throw(BoundsError(A, I))
@inline Base.checkbounds(::Type{Bool}, tgt::AreaTargets{NIons}, i, j) where {NIons} =
    0 < i <= NIons && 0 < j <= NIons && i != j
@inline Base.checkbounds(tgt::AreaTargets, i, j) =
    checkbounds(Bool, tgt, i, j) || throw_boundserror(tgt, (i, j))

@inline function pair_idx(NIons::Number, i, j)
    if i > j
        i, j = j, i
    end
    return (2 * NIons - i) * (i - 1) ÷ 2 + (j - i)
end
@inline pair_idx(tgt::AreaTargets{NIons}, i, j) where NIons = pair_idx(NIons, i, j)
function Base.Matrix(tgt::AreaTargets{NIons}) where {NIons}
    return Float64[tgt[i, j] for i in 1:NIons, j in 1:NIons]
end

Base.@propagate_inbounds function Base.getindex(tgt::AreaTargets, i, j)
    if i == j
        return 0.0
    end
    @boundscheck checkbounds(tgt, i, j)
    return @inbounds tgt.targets[pair_idx(tgt, i, j)]
end
Base.@propagate_inbounds function Base.setindex!(tgt::AreaTargets, v, i, j)
    @boundscheck checkbounds(tgt, i, j)
    @inbounds tgt.targets[pair_idx(tgt, i, j)] = v
end

Base.@propagate_inbounds function getweight(tgt::AreaTargets, i, j)
    @boundscheck checkbounds(tgt, i, j)
    return @inbounds tgt.weights[pair_idx(tgt, i, j)]
end
Base.@propagate_inbounds function setweight!(tgt::AreaTargets, v, i, j)
    @boundscheck checkbounds(tgt, i, j)
    @inbounds tgt.weights[pair_idx(tgt, i, j)] = v
end

@inline function (tgt::AreaTargets)(x, grads_out)
    NPairs = length(tgt.targets)
    res = 0.0
    has_grad = !isempty(grads_out)
    @inbounds @simd ivdep for i in 1:NPairs
        d = x[i] - tgt.targets[i]
        dw = d * tgt.weights[i]
        if has_grad
            grads_out[i] = 2 * dw
        end
        res = muladd(d, dw, res)
    end
    return res
end

mutable struct Kernel{NSeg,NModes,NIons,Omegas,Weights,PairBuff,ObjBuff}
    const ωs::Omegas
    const weights::Weights
    const pair_buffs::PairBuff
    const obj_args_buff::ObjBuff
    const obj_grad_buff::ObjBuff
    function Kernel{NSeg,NModes,NIons}(_ωs, bij, ηs) where {NSeg,NModes,NIons}
        ωs = SVector{NModes,Float64}(_ωs)
        NPairs = NIons * (NIons - 1) ÷ 2
        m_weights = MMatrix{NModes,NPairs,Float64}(undef)
        pair_idx = 0
        @inbounds for ion1 in 1:NIons - 1
            for ion2 in ion1 + 1:NIons
                pair_idx += 1
                nions = length(ηs)
                @simd ivdep for modei in 1:NModes
                    m_weights[modei, pair_idx] =
                        bij[modei, ion1] * bij[modei, ion2] * ηs[modei]^2
                end
            end
        end
        weights = SMatrix(m_weights)
        pair_buffs = MMatrix{4 * NSeg + 2,NPairs,Float64}(undef)
        obj_args_buff = MVector{NPairs,Float64}(undef)
        obj_grad_buff = MVector{NPairs,Float64}(undef)
        return new{NSeg,NModes,NIons,typeof(ωs),typeof(weights),
                   typeof(pair_buffs),typeof(obj_args_buff)}(ωs, weights, pair_buffs,
                                                             obj_args_buff,
                                                             obj_grad_buff)
    end
    function Kernel{NSeg}(ωs, bij, ηs) where {NSeg}
        NModes, NIons = size(bij)
        @assert length(ωs) == NModes
        @assert length(ηs) == NModes
        return Kernel{NSeg,NModes,NIons}(ωs, bij, ηs)
    end
end

@inline function _copy!(tgt, src, scale, keep)
    @inbounds @simd ivdep for i in 1:length(tgt)
        tgt[i] = dynamic(keep) ? muladd(src[i], scale, tgt[i]) : (src[i] * scale)
    end
end

@inline function _copy_grads!(grads_τs, grads_Ω1s, grads_Ω2s, grads_δs, pair_buff,
                              w, keep1, keep2, NSeg)
    @inbounds begin
        _copy!(grads_τs, @view(pair_buff[1:NSeg]), w, keep1)
        _copy!(grads_Ω1s, @view(pair_buff[NSeg + 1:2 * NSeg + 1]), w, keep1)
        _copy!(grads_Ω2s, @view(pair_buff[2 * NSeg + 2:3 * NSeg + 2]), w, keep2)
        _copy!(grads_δs, @view(pair_buff[3 * NSeg + 3:4 * NSeg + 2]), w, keep1)
    end
end

@inline function _evaluate_obj_args(kern::Kernel{NSeg,NModes,NIons},
                                    x, has_grad) where {NSeg,NModes,NIons}
    pair_idx = 0
    τs = @inbounds @view(x[1:NSeg])
    offsetδ = NSeg + NIons * (NSeg + 1)
    δs = @inbounds @view(x[offsetδ + 1:offsetδ + NSeg])
    @inbounds for ion1 in 1:NIons - 1
        offset1 = NSeg + (ion1 - 1) * (NSeg + 1)
        Ω1s = @view(x[offset1 + 1:offset1 + NSeg + 1])
        for ion2 in ion1 + 1:NIons
            pair_idx += 1
            offset2 = NSeg + (ion2 - 1) * (NSeg + 1)
            Ω2s = @view(x[offset2 + 1:offset2 + NSeg + 1])
            ws = @view(kern.weights[:, pair_idx])
            pair_buff = @view(kern.pair_buffs[:, pair_idx])
            if dynamic(has_grad)
                kern.obj_args_buff[pair_idx] =
                    enclosed_area_seq(τs, Ω1s, Ω2s, δs, kern.ωs, ws,
                                      @view(pair_buff[1:NSeg]),
                                      @view(pair_buff[NSeg + 1:2 * NSeg + 1]),
                                      @view(pair_buff[2 * NSeg + 2:3 * NSeg + 2]),
                                      @view(pair_buff[3 * NSeg + 3:4 * NSeg + 2]))
            else
                kern.obj_args_buff[pair_idx] =
                    enclosed_area_seq(τs, Ω1s, Ω2s, δs, kern.ωs, ws,
                                      (), (), (), ())
            end
        end
    end
end

@inline function _evaluate(kern::Kernel{NSeg,NModes,NIons}, @specialize(obj),
                           x, grads_out, has_grad) where {NSeg,NModes,NIons}
    _evaluate_obj_args(kern, x, has_grad)
    if !dynamic(has_grad)
        return obj(kern.obj_args_buff, ())
    end
    res = obj(kern.obj_args_buff, kern.obj_grad_buff)
    pair_idx = 0
    grads_τs = @inbounds @view(grads_out[1:NSeg])
    offsetδ = NSeg + NIons * (NSeg + 1)
    grads_δs = @inbounds @view(grads_out[offsetδ + 1:offsetδ + NSeg])
    @inbounds for ion1 in 1:NIons - 1
        offset1 = NSeg + (ion1 - 1) * (NSeg + 1)
        grads_Ω1s = @view(grads_out[offset1 + 1:offset1 + NSeg + 1])
        for ion2 in ion1 + 1:NIons
            pair_idx += 1
            offset2 = NSeg + (ion2 - 1) * (NSeg + 1)
            grads_Ω2s = @view(grads_out[offset2 + 1:offset2 + NSeg + 1])
            pair_buff = @view(kern.pair_buffs[:, pair_idx])
            grad_weight = kern.obj_grad_buff[pair_idx]
            if ion1 > 1
                _copy_grads!(grads_τs, grads_Ω1s, grads_Ω2s, grads_δs, pair_buff,
                             grad_weight, static(true), static(true), NSeg)
            elseif pair_idx > 1
                _copy_grads!(grads_τs, grads_Ω1s, grads_Ω2s, grads_δs, pair_buff,
                             grad_weight, static(true), static(false), NSeg)
            else
                _copy_grads!(grads_τs, grads_Ω1s, grads_Ω2s, grads_δs, pair_buff,
                             grad_weight, static(false), static(false), NSeg)
            end
        end
    end
    return res
end

(kern::Kernel)(@specialize(obj), x, grads_out) = if isempty(grads_out)
    return _evaluate(kern, obj, x, (), static(false))
else
    return _evaluate(kern, obj, x, grads_out, static(true))
end

function AreaTargets(kern::Kernel{NSeg,NModes,NIons}, x) where {NSeg,NModes,NIons}
    _evaluate_obj_args(kern, x, static(false))
    tgt = AreaTargets{NIons}()
    tgt.targets .= kern.obj_args_buff
    return tgt
end

end
