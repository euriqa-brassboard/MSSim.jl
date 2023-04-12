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

mutable struct SingleModeResult{T,A,CD,AG}
    τ::T
    area::A
    cumdis::CD
    area_mode::AG

    τ_grad::Utils.JaggedMatrix{T}
    area_grad::Utils.JaggedMatrix{A}
    cumdis_grad::Utils.JaggedMatrix{CD}
    area_mode_grad::Utils.JaggedMatrix{AG}
    function SingleModeResult{T,A,CD,AG}() where {T,A,CD,AG}
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

function compute_single_mode!(
    result::SingleModeResult{T,A,CD,AG},
    segments::AbstractVector{SD},
    buffer::SeqComputeBuffer{T},
    seg_grads::Union{_SGV{T,A,CD,AG},Nothing}=nothing) where SD <: SegData{T,A,CD,AG} where {T,A,CD,AG}

    nseg = length(segments)
    need_cumdis = !is_cumdis_dummy(SD)
    need_area_mode = !is_area_mode_dummy(SD)
    need_grads = seg_grads !== nothing
    resize!(buffer.dis_backward, need_grads || need_area_mode ? nseg : 0)
    resize!(buffer.τ_backward, need_grads && need_cumdis ? nseg : 0)
    resize!(buffer.disφ, need_area_mode ? nseg : 0)
    resize!(buffer.disφ_backward, need_grads && need_area_mode ? nseg : 0)
    if need_area_mode
        p_τ = zero(T)
        for i in 1:nseg
            seg = segments[i]
            buffer.disφ[i] = seg.area_mode.disδ + Utils.mulim(seg.area.dis * p_τ)
            p_τ += seg.τ
        end
    end
    if need_grads || need_area_mode
        p_τ = zero(T)
        p_dis = complex(zero(T))
        p_real_disδ = complex(zero(T))
        for i in nseg:-1:1
            seg = segments[i]
            if need_grads && need_cumdis
                buffer.τ_backward[i] = p_τ
            end
            buffer.dis_backward[i] = p_dis
            if need_grads && need_area_mode
                buffer.disφ_backward[i] = p_real_disδ
                p_real_disδ += buffer.disφ[i]
            end
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
            real_disδ = buffer.disφ[i]
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
            disφ_b = need_area_mode ? buffer.disφ_backward[i] : complex(zero(T))
            τ_b = need_cumdis ? buffer.τ_backward[i] : zero(T)

            for j in 1:nvar
                sg = seg_grad[j]

                τ_v = sg.τ
                τ_grad[j] = τ_v

                dis_v = sg.area.dis
                area_v = (imag(conj(p_dis) * dis_v) + imag(dis_b * conj(dis_v))
                          + sg.area.area)
                area_grad[j] = A(dis_v, area_v)

                if need_cumdis
                    cumdis_v = τ_v * p_dis + τ_b * dis_v + sg.cumdis.cumdis
                    cumdis_grad[j] = CD(cumdis_v)
                else
                    cumdis_grad[j] = CD(nothing)
                end

                if need_area_mode
                    disδ_v0 = sg.area_mode.disδ + Utils.mulim(dis_v * p_τ)
                    disδ_v = disδ_v0 + Utils.mulim(dis_b * τ_v)
                    # Note that the expression below isn't simply the derivative
                    # of the areaδ. This is because disδ depends on the sum of τ's
                    # so areaδ actually contains a thrird summation
                    # that breaks the change-of-summation-order trick that we've
                    # used for all other expressions.
                    areaδ_v = (sg.area_mode.areaδ +
                        imag(conj(p_dis) * disδ_v) +
                        imag(conj(dis_v) * disφ_b) +
                        τ_v * real(conj(dis_b) * seg.area.dis) +
                        imag(dis_b * conj(disδ_v0)) +
                        imag(dis_v * conj(p_real_disδ)))
                    area_mode_grad[j] = AG(disδ_v, areaδ_v)
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

mutable struct MultiModeResult{T,VCD,VDD,VAD}
    τ::T
    dis::Vector{Complex{T}}
    area::T
    cumdis::VCD # Vector
    disδ::VDD # Vector
    areaδ::VAD

    τ_grad::Utils.JaggedMatrix{T}
    dis_grad::Vector{Utils.JaggedMatrix{Complex{T}}}
    area_grad::Utils.JaggedMatrix{T}
    cumdis_grad::Vector{Utils.JaggedMatrix{Complex{T}}}
    disδ_grad::Vector{Utils.JaggedMatrix{Complex{T}}}
    areaδ_grad::Vector{Utils.JaggedMatrix{T}}

    function MultiModeResult{T}(::Val{need_cumdis},
                                ::Val{need_area_mode}) where {T,need_cumdis,
                                                              need_area_mode}
        VCD = need_cumdis ? Vector{Complex{T}} : Nothing
        VDD = need_area_mode ? Vector{Complex{T}} : Nothing
        VAD = need_area_mode ? Vector{T} : Nothing

        return new{T,VCD,VDD,VAD}(zero(T), Complex{T}[], zero(T),
                                  VCD(), VDD(), VAD(),
                                  Utils.JaggedMatrix{T}(),
                                  Utils.JaggedMatrix{Complex{T}}[],
                                  Utils.JaggedMatrix{T}(),
                                  Utils.JaggedMatrix{Complex{T}}[],
                                  Utils.JaggedMatrix{Complex{T}}[],
                                  Utils.JaggedMatrix{T}[])
    end
end

function _init_grads_vector(grads::Vector{Utils.JaggedMatrix{T}}, nmodes) where T
    resize!(grads, nmodes)
    for i in 1:nmodes
        if isassigned(grads, i)
            empty!(grads[i])
        else
            grads[i] = Utils.JaggedMatrix{T}()
        end
    end
    return
end

function (init_multi_mode_result!(result::MultiModeResult{T,VCD,VDD,AD},
                                  nmodes, ::Val{need_grad})
          where {T,VCD,VDD,AD,need_grad})

    need_cumdis = VCD !== Nothing
    need_area_mode = AD !== Nothing

    result.τ = zero(T)
    resize!(result.dis, nmodes)
    result.dis .= complex(zero(T))
    result.area = zero(T)
    if need_cumdis
        resize!(result.cumdis, nmodes)
        result.cumdis .= complex(zero(T))
    end
    if need_area_mode
        resize!(result.disδ, nmodes)
        result.disδ .= complex(zero(T))
        resize!(result.areaδ, nmodes)
        result.areaδ .= zero(T)
    end

    empty!(result.τ_grad)
    empty!(result.area_grad)
    if need_grad
        _init_grads_vector(result.dis_grad, nmodes)
        if need_cumdis
            _init_grads_vector(result.cumdis_grad, nmodes)
        end
        if need_area_mode
            _init_grads_vector(result.disδ_grad, nmodes)
            _init_grads_vector(result.areaδ_grad, nmodes)
        end
    else
        empty!(result.dis_grad)
        empty!(result.cumdis_grad)
        empty!(result.disδ_grad)
        empty!(result.areaδ_grad)
    end
end

end
