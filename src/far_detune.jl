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

Base.@propagate_inbounds function pair_idx(tgt::AreaTargets{NIons}, _i, _j) where NIons
    if _i > _j
        i, j = _j, _i
    else
        i, j = _i, _j
    end
    @boundscheck if i == j || j > NIons || i <= 0
        throw(BoundsError(tgt, (_i, _j)))
    end
    return (2 * NIons - i) * (i - 1) ÷ 2 + (j - i)
end

Base.@propagate_inbounds Base.getindex(tgt::AreaTargets, i, j) =
    tgt.targets[pair_idx(tgt, i, j)]
Base.@propagate_inbounds Base.setindex!(tgt::AreaTargets, v, i, j) =
    (tgt.targets[pair_idx(tgt, i, j)] = v)

Base.@propagate_inbounds getweight(tgt::AreaTargets, i, j) =
    tgt.weights[pair_idx(tgt, i, j)]
Base.@propagate_inbounds setweight!(tgt::AreaTargets, v, i, j) =
    (tgt.weights[pair_idx(tgt, i, j)] = v)

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

end
