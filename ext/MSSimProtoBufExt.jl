#

module MSSimProtoBufExt

import ProtoBuf as PB
import MSSim.Sequence as Seq

# PB.default_values(::Type{Seq.SolutionProperties}) = (;total_time = zero(Float64), modes = Vector{Float64}(), rdis = Vector{Float64}(), idis = Vector{Float64}(), rdisδ = Vector{Float64}(), idisδ = Vector{Float64}(), rcumdis = Vector{Float64}(), icumdis = Vector{Float64}(), area = Vector{Float64}(), areaδ = Vector{Float64}())
PB.field_numbers(::Type{Seq.SolutionProperties}) = (;total_time = 1, modes = 2, rdis = 3, idis = 4, rdisδ = 5, idisδ = 6, rcumdis = 7, icumdis = 8, area = 9, areaδ = 10)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Seq.SolutionProperties})
    total_time = zero(Float64)
    modes = PB.BufferedVector{Float64}()
    rdis = PB.BufferedVector{Float64}()
    idis = PB.BufferedVector{Float64}()
    rdisδ = PB.BufferedVector{Float64}()
    idisδ = PB.BufferedVector{Float64}()
    rcumdis = PB.BufferedVector{Float64}()
    icumdis = PB.BufferedVector{Float64}()
    area = PB.BufferedVector{Float64}()
    areaδ = PB.BufferedVector{Float64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            total_time = PB.decode(d, Float64)
        elseif field_number == 2
            PB.decode!(d, wire_type, modes)
        elseif field_number == 3
            PB.decode!(d, wire_type, rdis)
        elseif field_number == 4
            PB.decode!(d, wire_type, idis)
        elseif field_number == 5
            PB.decode!(d, wire_type, rdisδ)
        elseif field_number == 6
            PB.decode!(d, wire_type, idisδ)
        elseif field_number == 7
            PB.decode!(d, wire_type, rcumdis)
        elseif field_number == 8
            PB.decode!(d, wire_type, icumdis)
        elseif field_number == 9
            PB.decode!(d, wire_type, area)
        elseif field_number == 10
            PB.decode!(d, wire_type, areaδ)
        else
            PB.skip(d, wire_type)
        end
    end
    return Seq.SolutionProperties(total_time, modes[], complex.(rdis[], idis[]),
                                  complex.(rdisδ[], idisδ[]),
                                  complex.(rcumdis[], icumdis[]), area[], areaδ[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Seq.SolutionProperties)
    initpos = position(e.io)
    x.total_time !== zero(Float64) && PB.encode(e, 1, x.total_time)
    !isempty(x.modes) && PB.encode(e, 2, x.modes)
    if !isempty(x.dis)
        PB.encode(e, 3, real.(x.dis))
        PB.encode(e, 4, imag.(x.dis))
    end
    if !isempty(x.disδ)
        PB.encode(e, 5, real.(x.disδ))
        PB.encode(e, 6, imag.(x.disδ))
    end
    if !isempty(x.cumdis)
        PB.encode(e, 7, real.(x.cumdis))
        PB.encode(e, 8, imag.(x.cumdis))
    end
    !isempty(x.area) && PB.encode(e, 9, x.area)
    !isempty(x.areaδ) && PB.encode(e, 10, x.areaδ)
    return position(e.io) - initpos
end

struct DummyF64Vector <: AbstractVector{Float64}
    len::Int
end
Base.sizeof(v::DummyF64Vector) = v.len * sizeof(Float64)
function PB._encoded_size(x::Seq.SolutionProperties)
    encoded_size = 0
    x.total_time !== zero(Float64) && (encoded_size += PB._encoded_size(x.total_time, 1))
    !isempty(x.modes) && (encoded_size += PB._encoded_size(x.modes, 2))
    if !isempty(x.dis)
        dv = DummyF64Vector(length(x.dis))
        encoded_size += PB._encoded_size(dv, 3)
        encoded_size += PB._encoded_size(dv, 4)
    end
    if !isempty(x.disδ)
        dv = DummyF64Vector(length(x.disδ))
        encoded_size += PB._encoded_size(dv, 5)
        encoded_size += PB._encoded_size(dv, 6)
    end
    if !isempty(x.cumdis)
        dv = DummyF64Vector(length(x.cumdis))
        encoded_size += PB._encoded_size(dv, 7)
        encoded_size += PB._encoded_size(dv, 8)
    end
    !isempty(x.area) && (encoded_size += PB._encoded_size(x.area, 9))
    !isempty(x.areaδ) && (encoded_size += PB._encoded_size(x.areaδ, 10))
    return encoded_size
end

end
