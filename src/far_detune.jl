#

module FarDetune

using Static

macro accum_grad(keep, tgt, g, w)
    tgt = esc(tgt)
    g = esc(g)
    w = esc(w)
    quote
        if dynamic($(esc(keep)))
            $tgt = muladd($g, $w, $tgt)
        else
            $tgt = $g * $w
        end
    end
end

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
    res = @inbounds enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ - ωs[1], grad,
                                         weights[1]; keep_grad=keep_grad)
    @inbounds for i in 2:nmodes
        res += enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ - ωs[i], grad,
                                    weights[i]; keep_grad=keep_all_grad)
    end
    return res
end

end
