#

module FarDetune

@inline function enclosed_area_kernel(τ, Ω11, Ω12, Ω21, Ω22, δ, grad)
    δ⁻¹ = 1 / δ
    δ⁻¹_12 = δ⁻¹ / 12
    A = muladd(2, Ω21, Ω22)
    B = muladd(2, Ω22, Ω21)
    grad_τ = (A * Ω11 + B * Ω12) * δ⁻¹_12

    res = τ * grad_τ

    @inbounds if !isempty(grad)
        τδ⁻¹_12 = τ * δ⁻¹_12
        grad[1] = grad_τ
        grad[2] = τδ⁻¹_12 * A
        grad[3] = τδ⁻¹_12 * B
        grad[4] = τδ⁻¹_12 * muladd(2, Ω11, Ω12)
        grad[5] = τδ⁻¹_12 * muladd(2, Ω12, Ω11)
        grad[6] = -res * δ⁻¹
    end
    return res
end

end
