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

struct SegData{T,A,CD,AG,Ngrad}
    τ::T
    area::A
    cumdis::CD
    area_mode::AG

    area_grad::NTuple{Ngrad,A}
    cumdis_grad::NTuple{Ngrad,CD}
    area_mode_grad::NTuple{Ngrad,AG}
end
const SegDataNoGrad{T,A,CD,AG} = SegData{T,A,CD,AG,0}
is_cumdis_dummy(::Type{SegData{T,A,CD,AG,Ngrad}}) where {T,A,CD,AG,Ngrad} =
    is_dummy(CD)
is_area_mode_dummy(::Type{SegData{T,A,CD,AG,Ngrad}}) where {T,A,CD,AG,Ngrad} =
    is_dummy(AG)

mutable struct SeqResultData{T,A,CD,AG}
    τ::T
    area::A
    cumdis::CD
    area_mode::AG

    area_grad::Matrix{A}
    cumdis_grad::Matrix{CD}
    area_mode_grad::Matrix{AG}

    function SeqResultData{T,A,CD,AG}(nseg, ngrad) where {T,A,CD,AG}
        return new(zero(T), A(), CD(), AG(),
                   Matrix{A}(undef, nseg, ngrad),
                   Matrix{CD}(undef, nseg, ngrad),
                   Matrix{AG}(undef, nseg, ngrad))
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

function compute_sequence!(
    result::SeqResultData{T,A,CD,AG},
    segments::AbstractVector{SD},
    buffer::SeqComputeBuffer{T}) where SD <: SegDataNoGrad{T,A,CD,AG} where {T,A,CD,AG}

    nseg = length(segments)
    need_cumdis = !is_cumdis_dummy(SD)
    need_area_mode = !is_area_mode_dummy(SD)
    if need_area_mode
        resize!(buffer.dis_backward, nseg)
    else
        resize!(buffer.dis_backward, 0)
    end
    resize!(buffer.τ_backward, 0)
    resize!(buffer.τ_forward, 0)
    resize!(buffer.disφ_backward, 0)
    if need_area_mode
        p_dis = complex(zero(T))
        for i in nseg:-1:1
            seg = segments[i]
            buffer.dis_backward[i] = p_dis
            p_dis += seg.area.dis
        end
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
        np_area += conj(p_dis) * seg.area.dis + seg.area.area
        if need_cumdis
            np_cumdis += p_dis * seg.τ + seg.cumdis.cumdis
        end
        if need_area_mode
            real_disδ = seg.area_mode.disδ + Utils.mulim(seg.area.dis * p_τ)
            np_real_disδ += real_disδ
            np_areaδ += (conj(p_dis) * real_disδ +
                buffer.dis_backward[i] * conj(real_disδ) + seg.area_mode.areaδ)
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
