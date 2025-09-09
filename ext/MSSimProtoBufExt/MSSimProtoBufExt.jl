#

module MSSimProtoBufExt

import ProtoBuf as PB
import MSSim.Sequence as Seq

include("ms_solutions_pb.jl")

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Seq.SolutionProperties})
    props = PB.decode(d, ms_solutions_pb.SolutionProperties)
    return Seq.SolutionProperties(props.total_time, props.modes,
                                  complex.(props.rdis, props.idis),
                                  complex.(props.rdis_det, props.idis_det),
                                  complex.(props.rcumdis, props.icumdis),
                                  props.area, props.area_det)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Seq.SolutionProperties)
    return PB.encode(e, ms_solutions_pb.SolutionProperties(
        x.total_time, x.modes, real.(x.dis), imag.(x.dis),
        real.(x.disδ), imag.(x.disδ), real.(x.cumdis), imag.(x.cumdis),
        x.area, x.areaδ))
end

end
