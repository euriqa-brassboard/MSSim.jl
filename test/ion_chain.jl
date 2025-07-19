#!/usr/bin/julia

using Test

using MSSim
const IC = MSSim.IonChain

@testset "Harmonic potential" begin
    ions = IC.simple_ions(2)
    @test ions[1] === ions[2]
    @test ions[1].charge == 1
    @test ions[1].mass == 1

    coeffs, func = IC.poly_function(Val(2))
    model = IC.AxialModel(ions, func)
    IC.set_init_pos!(model, 1, -1)
    IC.set_init_pos!(model, 2, 1)
    for x1 in range(-5, 5, 101)
        coeffs[1] = x1
        for x2 in range(0.1, 5, 50)
            coeffs[2] = x2
            pos = IC.optimize!(model)
            @test pos[1] ≈ -x1 / 2 / x2 - 1 / 2 / cbrt(x2) atol=1e-3 rtol=1e-3
            @test pos[2] ≈ -x1 / 2 / x2 + 1 / 2 / cbrt(x2) atol=1e-3 rtol=1e-3

            freqs, vecs = IC.axial_modes(ions, pos, func)
            @test freqs[1] ≈ sqrt(x2 * 2)
            @test freqs[2] ≈ sqrt(x2 * 2 * 3) atol=2e-3 rtol=2e-3
            @test vecs[1, 1] ≈ vecs[2, 1]
            @test vecs[1, 2] ≈ -vecs[2, 2]

            freqs, vecs = IC.radial_modes(ions, pos, IC.Function1D(x->3.0))
            @test freqs[2]^2 - freqs[1] * abs(freqs[1]) ≈ x2 * 2 atol=3e-3 rtol=3e-3
            @test freqs[2] ≈ sqrt(3.0)
            @test vecs[1, 1] ≈ -vecs[2, 1]
            @test vecs[1, 2] ≈ vecs[2, 2]
        end
    end
end
