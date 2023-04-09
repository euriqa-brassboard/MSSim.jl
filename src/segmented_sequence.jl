#!/usr/bin/julia

module SegSeq

using ..Utils

is_dummy(::Type{T}) where T = false

struct AreaData{T}
    dis::Complex{T}
    area::T
    AreaData{T}(dis, area) where T = new(dis, area)
    AreaData{T}() where T = new(zero(T), zero(T))
end

struct CumDisData{T,CT}
    cumdis::CT
    CumDisData{T,CT}(cumdis) where {T,CT} = new(cumdis)
    CumDisData{T,CT}() where {T,CT} = new(zero(T))
    CumDisData{Nothing,Nothing}() = new(nothing)
end
const DummyCumDisData = CumDisData{Nothing,Nothing}
is_dummy(::Type{DummyCumDisData}) = true

struct AreaModeData{T,CT}
    disδ::CT
    areaδ::T
    AreaModeData{T,CT}(disδ, areaδ) where {T,CT} = new(disδ, areaδ)
    AreaModeData{T,CT}() where {T,CT} = new(zero(T), zero(T))
    AreaModeData{Nothing,Nothing}() = new(nothing, nothing)
end
const DummyAreaModeData = AreaModeData{Nothing,Nothing}
is_dummy(::Type{DummyAreaModeData}) = true

struct SegData{T,A,CD,AG}
    τ::T
    area::A
    cumdis::CD
    area_mode::AG
end
is_cumdis_dummy(::Type{SegData{T,A,CD,AG}}) where {T,A,CD,AG} =
    is_dummy(CD)
is_area_mode_dummy(::Type{SegData{T,A,CD,AG}}) where {T,A,CD,AG} =
    is_dummy(AG)

const _SGS{T,A,CD,AG} = AbstractVector{SegData{T,A,CD,AG}}
const _SGV{T,A,CD,AG} = AbstractVector{SGS} where SGS <: _SGS{T,A,CD,AG}

mutable struct SeqResultData{T,A,CD,AG}
    τ::T
    area::A
    cumdis::CD
    area_mode::AG

    τ_grad::Utils.JaggedMatrix{T}
    area_grad::Utils.JaggedMatrix{A}
    cumdis_grad::Utils.JaggedMatrix{CD}
    area_mode_grad::Utils.JaggedMatrix{AG}
    function SeqResultData{T,A,CD,AG}() where {T,A,CD,AG}
        return new(zero(T), A(), CD(), AG(),
                   Utils.JaggedMatrix{T}(), Utils.JaggedMatrix{A}(),
                   Utils.JaggedMatrix{CD}(), Utils.JaggedMatrix{AG}())
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
    # Partial sum of the first n (0 to N-1) time step lengths.
    # Used to compute: disφ_backward
    τ_forward::Vector{T}
    # Partial sum of the last n (0 to N-1) gradient of displacement w.r.t. detuning.
    # Used to compute:
    # * Gradient of area w.r.t. parameters and detuning
    disφ_backward::Vector{Complex{T}}
    function SeqComputeBuffer{T}() where T
        return new(Complex{T}[], T[], T[], Complex{T}[])
    end
end

# TODO gradients
function compute_sequence!(
    result::SeqResultData{T,A,CD,AG},
    segments::AbstractVector{SD},
    buffer::SeqComputeBuffer{T},
    seg_grads::Union{_SGV{T,A,CD,AG},Nothing}=nothing) where SD <: SegData{T,A,CD,AG} where {T,A,CD,AG}

    nseg = length(segments)
    need_cumdis = !is_cumdis_dummy(SD)
    need_area_mode = !is_area_mode_dummy(SD)
    need_grads = seg_grads !== nothing
    resize!(buffer.dis_backward, need_grads || need_area_mode ? nseg : 0)
    resize!(buffer.τ_backward, need_grads && need_cumdis ? nseg : 0)
    resize!(buffer.τ_forward, need_grads && need_area_mode ? nseg : 0)
    resize!(buffer.disφ_backward, need_grads && need_area_mode ? nseg : 0)
    if need_grads && need_area_mode
        p_τ = zero(T)
        for i in 1:nseg
            seg = segments[i]
            buffer.τ_forward[i] = p_τ
            p_τ += seg.τ
        end
    end
    if need_grads || need_area_mode
        p_τ = zero(T)
        p_dis = complex(zero(T))
        for i in nseg:-1:1
            seg = segments[i]
            if need_grads && need_cumdis
                buffer.τ_backward[i] = p_τ
            end
            buffer.dis_backward[i] = p_dis
            p_τ += seg.τ
            p_dis += seg.area.dis
        end
    end

    if need_grads
        @assert length(seg_grads) == nseg
        resize!(result.τ_grad, seg_grads)
        resize!(result.area_grad, seg_grads)
        resize!(result.cumdis_grad, seg_grads)
        resize!(result.area_mode_grad, seg_grads)
    else
        empty!(result.τ_grad)
        empty!(result.area_grad)
        empty!(result.cumdis_grad)
        empty!(result.area_mode_grad)
    end

    p_τ = zero(T)
    p_dis = complex(zero(T))
    p_area = zero(T)
    p_cumdis = complex(zero(T))
    p_real_disδ = complex(zero(T))
    p_areaδ = zero(T)
    for i in 1:nseg
        seg = segments[i]
        np_τ = p_τ
        np_dis = p_dis
        np_area = p_area
        np_cumdis = p_cumdis
        np_real_disδ = p_real_disδ
        np_areaδ = p_areaδ

        np_τ += seg.τ
        np_dis += seg.area.dis
        np_area += imag(conj(p_dis) * seg.area.dis) + seg.area.area
        if need_cumdis
            np_cumdis += p_dis * seg.τ + seg.cumdis.cumdis
        end
        if need_area_mode
            real_disδ = seg.area_mode.disδ + Utils.mulim(seg.area.dis * p_τ)
            np_real_disδ += real_disδ
            np_areaδ += (imag(conj(p_dis) * real_disδ) +
                imag(buffer.dis_backward[i] * conj(real_disδ)) + seg.area_mode.areaδ)
        end
        @inline if need_grads
            seg_grad = seg_grads[i]

            τ_grad = result.τ_grad[i]
            area_grad = result.area_grad[i]
            cumdis_grad = result.cumdis_grad[i]
            area_mode_grad = result.area_mode_grad[i]
            nvar = length(seg_grad)

            dis_b = buffer.dis_backward[i]
            τ_b = buffer.τ_backward[i]

            for j in 1:nvar
                τ_v = seg_grad[j].τ
                τ_grad[j] = τ_v
                dis_v = seg_grad[j].area.dis
                area_v = (imag(conj(p_dis) * dis_v) + imag(dis_b * conj(dis_v))
                          + seg_grad[j].area.area)
                area_grad[j] = A(dis_v, area_v)
                if need_cumdis
                    cumdis_v = τ_b * dis_v + seg_grad[j].cumdis.cumdis
                    cumdis_grad[j] = CD(cumdis_v)
                else
                    cumdis_grad[j] = CD(nothing)
                end
                if need_area_mode
                    # TODO
                    area_mode_grad[j] = AG(0, 0)
                else
                    area_mode_grad[j] = AG(nothing, nothing)
                end
            end
        end

        p_τ = np_τ
        p_dis = np_dis
        p_area = np_area
        p_cumdis = np_cumdis
        p_real_disδ = np_real_disδ
        p_areaδ = np_areaδ
    end

    result.τ = p_τ
    result.area = A(p_dis, p_area)
    if need_cumdis
        result.cumdis = CD(p_cumdis)
    end
    if need_area_mode
        result.area_mode = AG(p_real_disδ, p_areaδ)
    end
    return
end

end
