#!/usr/bin/julia

module SegSeq

# Common code to compute a MS gate that consists of many segments
# This includes the representation of segmented sequence,
# and the code to combine them to compute the properties of the whole sequence.

using ..Utils

# Sequence segment representation
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

    const τ_grad::Utils.JaggedMatrix{T}
    const area_grad::Utils.JaggedMatrix{A}
    const cumdis_grad::Utils.JaggedMatrix{CD}
    const area_mode_grad::Utils.JaggedMatrix{AG}
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

    buffer_dis_backward = buffer.dis_backward
    buffer_τ_backward = buffer.τ_backward
    buffer_disφ = buffer.disφ
    buffer_disφ_backward = buffer.disφ_backward

    resize!(buffer_dis_backward, need_grads || need_area_mode ? nseg : 0)
    resize!(buffer_τ_backward, need_grads && need_cumdis ? nseg : 0)
    resize!(buffer_disφ, need_area_mode ? nseg : 0)
    resize!(buffer_disφ_backward, need_grads && need_area_mode ? nseg : 0)
    @inbounds if need_area_mode
        p_τ = zero(T)
        for i in 1:nseg
            seg = segments[i]
            buffer_disφ[i] = muladd(Utils.mulim(seg.area.dis), p_τ, seg.area_mode.disδ)
            p_τ += seg.τ
        end
    end
    @inbounds if need_grads || need_area_mode
        p_τ = zero(T)
        p_dis = complex(zero(T))
        p_real_disδ = complex(zero(T))
        for i in nseg:-1:1
            seg = segments[i]
            if need_grads && need_cumdis
                buffer_τ_backward[i] = p_τ
            end
            buffer_dis_backward[i] = p_dis
            if need_grads && need_area_mode
                buffer_disφ_backward[i] = p_real_disδ
                p_real_disδ += buffer_disφ[i]
            end
            p_τ += seg.τ
            p_dis += seg.area.dis
        end
    end

    result_τ_grad = result.τ_grad
    result_area_grad = result.area_grad
    result_cumdis_grad = result.cumdis_grad
    result_area_mode_grad = result.area_mode_grad
    if need_grads
        @assert length(seg_grads) == nseg
        resize!(result_τ_grad, seg_grads)
        resize!(result_area_grad, seg_grads)
        resize!(result_cumdis_grad, seg_grads)
        resize!(result_area_mode_grad, seg_grads)
    else
        empty!(result_τ_grad)
        empty!(result_area_grad)
        empty!(result_cumdis_grad)
        empty!(result_area_mode_grad)
    end

    p_τ = zero(T)
    p_dis = complex(zero(T))
    p_area = zero(T)
    p_cumdis = complex(zero(T))
    p_real_disδ = complex(zero(T))
    p_areaδ = zero(T)
    @inbounds for i in 1:nseg
        seg = segments[i]
        np_τ = p_τ
        np_dis = p_dis
        np_area = p_area
        np_cumdis = p_cumdis
        np_real_disδ = p_real_disδ
        np_areaδ = p_areaδ

        np_τ += seg.τ
        np_dis += seg.area.dis
        np_area += muladd(real(p_dis), imag(seg.area.dis),
                          muladd(-imag(p_dis), real(seg.area.dis), seg.area.area))
        if need_cumdis
            np_cumdis += muladd(p_dis, seg.τ, seg.cumdis.cumdis)
        end
        if need_area_mode
            real_disδ = buffer_disφ[i]
            np_real_disδ += real_disδ
            dis_b = buffer_dis_backward[i]
            np_areaδ += muladd(real(p_dis) - real(dis_b), imag(real_disδ),
                                muladd(imag(dis_b) - imag(p_dis), real(real_disδ),
                                       seg.area_mode.areaδ))
        end
        @inline if need_grads
            seg_grad = seg_grads[i]

            τ_grad = result_τ_grad[i]
            area_grad = result_area_grad[i]
            cumdis_grad = result_cumdis_grad[i]
            area_mode_grad = result_area_mode_grad[i]
            nvar = length(seg_grad)

            dis_b = buffer_dis_backward[i]
            disφ_b = need_area_mode ? buffer_disφ_backward[i] : complex(zero(T))
            τ_b = need_cumdis ? buffer_τ_backward[i] : zero(T)

            for j in 1:nvar
                sg = seg_grad[j]

                τ_v = sg.τ
                τ_grad[j] = τ_v

                dis_v = sg.area.dis
                area_v = muladd(real(p_dis) - real(dis_b), imag(dis_v),
                                muladd(imag(dis_b) - imag(p_dis), real(dis_v),
                                       sg.area.area))
                area_grad[j] = A(dis_v, area_v)

                if need_cumdis
                    cumdis_v = muladd(τ_v, p_dis, muladd(τ_b, dis_v, sg.cumdis.cumdis))
                    cumdis_grad[j] = CD(cumdis_v)
                else
                    cumdis_grad[j] = CD(nothing)
                end

                if need_area_mode
                    disδ_v0 = muladd(Utils.mulim(dis_v), p_τ, sg.area_mode.disδ)
                    disδ_v = muladd(Utils.mulim(dis_b), τ_v, disδ_v0)
                    # Note that the expression below isn't simply the derivative
                    # of the areaδ. This is because disδ depends on the sum of τ's
                    # so areaδ actually contains a third summation
                    # that breaks the change-of-summation-order trick that we've
                    # used for all other expressions.
                    areaδ_v =
                        muladd(
                            imag(dis_b), muladd(τ_v, imag(seg.area.dis), real(disδ_v0)),
                            muladd(
                                real(dis_b), muladd(τ_v, real(seg.area.dis),
                                                    -imag(disδ_v0)),
                                muladd(
                                    -imag(p_dis), real(disδ_v),
                                    muladd(
                                        real(p_dis), imag(disδ_v),
                                        muladd(real(dis_v),
                                               imag(disφ_b) - imag(p_real_disδ),
                                               muladd(imag(dis_v),
                                                      real(p_real_disδ) - real(disφ_b),
                                                      sg.area_mode.areaδ))))))
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

# Note that we only provide a data structure to store the result here.
# The combination of multiple modes into a final result is both simple enough
# and segment-specific enough that it's easier to just implement it
# by the user.
# If we ever come to a point where there are multiple users we can try to combine them
# together...
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
    @inbounds for i in 1:nmodes
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
