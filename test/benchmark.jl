#!/usr/bin/julia

using MSSim
using BenchmarkTools

for (cumdis, area_mode, grad) in Iterators.product((false, true), (false, true),
                                                   (false, true))
    @show (cumdis, area_mode, grad)
    sys = MSSim.SymLinear.System{Float64}(Val(cumdis), Val(area_mode), Val(grad))

    for i in 1:50
        push!(sys.modes, MSSim.SymLinear.Mode(i * 10.0, 1.0, 1.0))
    end

    for i in 1:100
        push!(sys.pulses, MSSim.SymLinear.Pulse{Float64}(10, 2, 3.4, 0, 5.0 * i))
    end

    display(@benchmark MSSim.SymLinear.compute!($(sys)))
end
