#!/usr/bin/julia

module SegSeq

# Common code to compute a MS gate that consists of many segments
# This includes the representation of segmented sequence,
# and the code to combine them to compute the properties of the whole sequence.

using ..Utils

_empty(::Type{T}) where T = T()
_empty(::Type{T}) where T<:Number = zero(T)
Base.@assume_effects :total _stype(T, b::Bool) = b ? T : Nothing

struct ValueMask
    τ::Bool
    dis::Bool
    area::Bool
    cumdis::Bool
    disδ::Bool
    areaδ::Bool
end
Base.zero(::Type{ValueMask}) = ValueMask(false, false, false, false, false, false)
const mask_full = ValueMask(true, true, true, true, true, true)
const mask_allδ = ValueMask(true, true, true, false, true, true)

# Sequence segment representation
struct SegData{T,TT,D,A,CD,DG,AG}
    τ::TT
    dis::D
    area::A
    cumdis::CD
    disδ::DG
    areaδ::AG
    function SegData{T,TT,D,A,CD,DG,AG}(τ=_empty(TT), dis=_empty(D), area=_empty(A),
                                        cumdis=_empty(CD), disδ=_empty(DG),
                                        areaδ=_empty(AG)) where {T,TT,D,A,CD,DG,AG}
        return new{T,TT,D,A,CD,DG,AG}(τ, dis, area, cumdis, disδ, areaδ)
    end
    function SegData{T}(τ::TT, dis::D, area::A, cumdis::CD, disδ::DG,
                        areaδ::AG) where {T,TT,D,A,CD,DG,AG}
        return new{T,TT,D,A,CD,DG,AG}(τ, dis, area, cumdis, disδ, areaδ)
    end
end
@inline function SegData(T, mask::ValueMask)
    CT = Complex{T}
    return SegData{T,_stype(T, mask.τ),_stype(CT, mask.dis),_stype(T, mask.area),
                   _stype(CT, mask.cumdis),_stype(CT, mask.disδ),
                   _stype(T, mask.areaδ)}
end
@inline value_mask(::Type{SegData{T,TT,D,A,CD,DG,AG}}) where {T,TT,D,A,CD,DG,AG} =
    ValueMask(TT !== Nothing, D !== Nothing, A !== Nothing,
              CD !== Nothing, DG !== Nothing, AG !== Nothing)
@inline value_mask(sd::SegData) = value_mask(typeof(sd))

mutable struct SingleModeResult{T,SDV<:SegData,SDG<:SegData}
    val::SDV
    const grad::Utils.JaggedMatrix{SDG}
    function SingleModeResult{T}(::Val{maskv}, ::Val{maskg}) where {T,maskv,maskg}
        SDV = SegData(T, maskv)
        SDG = SegData(T, maskg)
        return new{T,SDV,SDG}(SDV(), Utils.JaggedMatrix{SDG}())
    end
end

struct SeqComputeBuffer{T}
    # Partial sum of the last n (0 to N-1) displacement.
    # Used to compute:
    # * Gradient of area w.r.t. detuning
    # * Gradient of area w.r.t. parameters
    # * Gradient of area w.r.t. parameters and detuning
    dis_backward::Vector{Complex{T}}
    # Partial sum of the last n (0 to N-1) time step lengths.
    # Used to compute:
    # * Gradient of cumulative displacement w.r.t. parameters
    τ_backward::Vector{T}
    # Gradient of displacement w.r.t. detuning including the effect of the
    # phase accumulation in the previous segments due to detuning change.
    # Used to compute: disφ_backward
    disφ::Vector{Complex{T}}
    # Partial sum of the last n (0 to N-1) gradient of displacement w.r.t. detuning.
    # Used to compute:
    # * Gradient of area w.r.t. parameters and detuning
    disφ_backward::Vector{Complex{T}}
    function SeqComputeBuffer{T}() where T
        return new(Complex{T}[], T[], Complex{T}[], Complex{T}[])
    end
end

Base.@assume_effects :total function _check_mask(maskv, maskg, has_τ_grad)
    disφ_backward = maskg.areaδ
    buffer_mask = (disφ=maskv.disδ || maskv.areaδ || disφ_backward,
                   τ_backward=maskg.cumdis,
                   dis_backward=maskv.areaδ || maskg.area || maskg.disδ || maskg.areaδ,
                   disφ_backward=disφ_backward)

    if buffer_mask.disφ
        (maskv.τ && maskv.dis && maskv.disδ) || (return false, buffer_mask)
    end
    if buffer_mask.τ_backward
        maskv.τ || (return false, buffer_mask)
    end
    if buffer_mask.dis_backward
        maskv.dis || (return false, buffer_mask)
    end

    if maskv.area
        maskv.dis || (return false, buffer_mask)
    end
    if maskv.cumdis
        (maskv.τ && maskv.dis) || (return false, buffer_mask)
    end
    if maskv.areaδ
        maskv.dis || (return false, buffer_mask)
    end

    maskg_τ = maskg.τ || !has_τ_grad

    if maskg.area
        (maskv.dis && maskg.dis) || (return false, buffer_mask)
    end
    if maskg.cumdis
        (maskv.dis && maskg_τ && maskg.dis) || (return false, buffer_mask)
    end
    if maskg.disδ
        (maskv.τ && maskg.dis && maskg_τ) || (return false, buffer_mask)
    end
    if maskg.areaδ
        (maskv.dis && maskv.disδ &&
            maskg_τ && maskg.disδ && maskg.dis) || (return false, buffer_mask)
    end
    return true, buffer_mask
end

const _SGS{SD} = AbstractVector{SD}
const _SGV{SD} = AbstractVector{SGS} where SGS <: _SGS{SD}

function compute_single_mode!(
    result::SingleModeResult{T,SDV,SDG},
    segments::AbstractVector{SDV},
    buffer::SeqComputeBuffer{T},
    seg_grads::Union{_SGV{SDG},Nothing}=nothing,
    ::Val{has_τ_grad}=Val(true),
    ::Val{res_resized}=Val(false)) where {SDV<:SegData{T},SDG<:SegData{T}} where {T,has_τ_grad,res_resized}

    maskv = value_mask(SDV)
    maskg = seg_grads === nothing ? zero(ValueMask) : value_mask(SDG)
    mask_compatible, buffer_mask = _check_mask(maskv, maskg, has_τ_grad)
    if !mask_compatible
        error("Incompatible value mask: $maskv, $maskg")
    end

    nseg = length(segments)
    need_grads = maskg !== zero(ValueMask)

    buffer_dis_backward = buffer.dis_backward
    buffer_τ_backward = buffer.τ_backward
    buffer_disφ = buffer.disφ
    buffer_disφ_backward = buffer.disφ_backward

    resize!(buffer_dis_backward, buffer_mask.dis_backward ? nseg : 0)
    resize!(buffer_τ_backward, buffer_mask.τ_backward ? nseg : 0)
    resize!(buffer_disφ, buffer_mask.disφ ? nseg : 0)
    resize!(buffer_disφ_backward, buffer_mask.disφ_backward ? nseg : 0)
    @inbounds if buffer_mask.disφ
        p_τ = zero(T)
        for i in 1:nseg
            seg = segments[i]
            buffer_disφ[i] = muladd(Utils.mulim(seg.dis), p_τ, seg.disδ)
            p_τ += seg.τ
        end
    end
    @inbounds if (buffer_mask.τ_backward || buffer_mask.dis_backward ||
        buffer_mask.disφ_backward)

        p_τ = zero(T)
        p_dis = complex(zero(T))
        p_real_disδ = complex(zero(T))
        for i in nseg:-1:1
            seg = segments[i]
            if buffer_mask.τ_backward
                buffer_τ_backward[i] = p_τ
                p_τ += seg.τ
            end
            if buffer_mask.dis_backward
                buffer_dis_backward[i] = p_dis
                p_dis += seg.dis
            end
            if buffer_mask.disφ_backward
                buffer_disφ_backward[i] = p_real_disδ
                p_real_disδ += buffer_disφ[i]
            end
        end
    end

    result_grad = result.grad
    if !res_resized
        if need_grads
            @assert length(seg_grads) == nseg
            resize!(result_grad, seg_grads)
        else
            empty!(result_grad)
        end
    end

    p_τ = maskv.τ ? zero(T) : nothing
    p_dis = maskv.dis ? complex(zero(T)) : nothing
    p_area = maskv.area ? zero(T) : nothing
    p_cumdis = maskv.cumdis ? complex(zero(T)) : nothing
    p_real_disδ = maskv.disδ ? complex(zero(T)) : nothing
    p_areaδ = maskv.areaδ ? zero(T) : nothing
    @inbounds for i in 1:nseg
        seg = segments[i]

        if maskv.τ
            np_τ = p_τ + seg.τ
        end
        if maskv.dis
            np_dis = p_dis + seg.dis
        end
        if maskv.area
            np_area = p_area + muladd(real(p_dis), imag(seg.dis),
                                      muladd(-imag(p_dis), real(seg.dis), seg.area))
        end
        if maskv.cumdis
            np_cumdis = p_cumdis + muladd(p_dis, seg.τ, seg.cumdis)
        end
        if maskv.disδ
            real_disδ = buffer_disφ[i]
            np_real_disδ = p_real_disδ + real_disδ
            if maskv.areaδ
                dis_b = buffer_dis_backward[i]
                np_areaδ = muladd(real(p_dis) - real(dis_b), imag(real_disδ),
                                   muladd(imag(dis_b) - imag(p_dis), real(real_disδ),
                                          seg.areaδ)) + p_areaδ
            end
        end
        @inline if need_grads
            seg_grad = seg_grads[i]

            r_grad = result_grad[i]
            nvar = length(seg_grad)

            dis_b = (buffer_mask.dis_backward ?
                buffer_dis_backward[i] : complex(zero(T)))
            disφ_b = (buffer_mask.disφ_backward ?
                buffer_disφ_backward[i] : complex(zero(T)))
            τ_b = buffer_mask.τ_backward ? buffer_τ_backward[i] : zero(T)

            for j in 1:nvar
                sg = seg_grad[j]

                τ_v = has_τ_grad ? sg.τ : Utils.Zero()

                dis_v = sg.dis
                if maskg.area
                    area_v = muladd(real(p_dis) - real(dis_b), imag(dis_v),
                                    muladd(imag(dis_b) - imag(p_dis), real(dis_v),
                                           sg.area))
                else
                    area_v = nothing
                end

                if maskg.cumdis
                    cumdis_v = muladd(τ_v, p_dis, muladd(τ_b, dis_v, sg.cumdis))
                else
                    cumdis_v = nothing
                end

                if maskg.disδ
                    disδ_v0 = muladd(Utils.mulim(dis_v), p_τ, sg.disδ)
                    disδ_v = muladd(Utils.mulim(dis_b), τ_v, disδ_v0)
                else
                    disδ_v = nothing
                end
                if maskg.areaδ
                    # Note that the expression below isn't simply the derivative
                    # of the areaδ. This is because disδ depends on the sum of τ's
                    # so areaδ actually contains a third summation
                    # that breaks the change-of-summation-order trick that we've
                    # used for all other expressions.
                    areaδ_v =
                        muladd(
                            imag(dis_b), muladd(τ_v, imag(seg.dis), real(disδ_v0)),
                            muladd(
                                real(dis_b), muladd(τ_v, real(seg.dis),
                                                    -imag(disδ_v0)),
                                muladd(
                                    -imag(p_dis), real(disδ_v),
                                    muladd(
                                        real(p_dis), imag(disδ_v),
                                        muladd(real(dis_v),
                                               imag(disφ_b) - imag(p_real_disδ),
                                               muladd(imag(dis_v),
                                                      real(p_real_disδ) - real(disφ_b),
                                                      sg.areaδ))))))
                else
                    areaδ_v = nothing
                end
                r_grad[j] = SDG(has_τ_grad ? τ_v : (maskg.τ ? zero(T) : nothing),
                                dis_v, area_v, cumdis_v, disδ_v, areaδ_v)
            end
        end

        maskv.τ && (p_τ = np_τ)
        maskv.dis && (p_dis = np_dis)
        maskv.area && (p_area = np_area)
        maskv.cumdis && (p_cumdis = np_cumdis)
        maskv.disδ && (p_real_disδ = np_real_disδ)
        maskv.areaδ && (p_areaδ = np_areaδ)
    end

    result.val = SDV(p_τ, p_dis, p_area, p_cumdis, p_real_disδ, p_areaδ)
    return
end

end
