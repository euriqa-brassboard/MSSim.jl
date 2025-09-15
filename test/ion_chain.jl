#!/usr/bin/julia

using Test
using JuMP
using NLopt
using StaticArrays

using MSSim
const IC = MSSim.IonChain

@testset "Harmonic potential" begin
    ions = IC.simple_ions(2)
    @test ions[1] === ions[2]
    @test ions[1].charge == 1
    @test ions[1].mass == 1

    ions2 = [IC.IonInfo(1, 2), IC.IonInfo(1, 2)]
    @test ions2[1] === ions2[2]
    @test ions2[1].charge == 1
    @test ions2[1].mass == 2

    coeffs_dc, func_dc3 = IC.poly_function(Val(2))
    func_dc2 = IC.Function1D(func_dc3.f, func_dc3.∇f)
    func_dc1 = IC.Function1D(func_dc3.f)

    coeffs_rf, func_rf3 = IC.poly_function(Val(2))
    func_rf2 = IC.Function1D(func_rf3.f, func_rf3.∇f)
    func_rf1 = IC.Function1D(func_rf3.f)

    function check_harmonic(mass, func_dc, func_rf,
                            scale_axial_dc, scale_axial_rf,
                            scale_radial_dc, scale_radial_rf)
        ions = [IC.IonInfo(1, mass), IC.IonInfo(1, mass)]
        func_rf = scale_axial_rf == 0 ? nothing : func_rf
        model = IC.AxialModel(ions, func_dc, func_rf)

        pscale_axial = scale_axial_dc + scale_axial_rf / mass^2

        IC.set_init_pos!(model, 1, -1)
        IC.set_init_pos!(model, 2, 1)
        for x1 in range(-5, 5, 21)
            coeffs_dc[1] = x1 * scale_axial_dc
            coeffs_rf[1] = x1 * scale_axial_rf
            real_x1 = pscale_axial * x1
            for x2 in range(0.1, 5, 16)
                coeffs_dc[2] = x2 * scale_axial_dc
                coeffs_rf[2] = x2 * scale_axial_rf
                real_x2 = pscale_axial * x2
                pos = IC.optimize!(model)
                @test pos[1] ≈ -real_x1 / 2 / real_x2 - 1 / 2 / cbrt(real_x2) atol=1e-3 rtol=1e-3
                @test pos[2] ≈ -real_x1 / 2 / real_x2 + 1 / 2 / cbrt(real_x2) atol=1e-3 rtol=1e-3

                freqs, vecs = IC.axial_modes(ions, pos, func_dc, func_rf)
                @test freqs[1] ≈ sqrt(real_x2 * 2) / sqrt(mass)
                @test freqs[2] ≈ sqrt(real_x2 * 2 * 3) / sqrt(mass) atol=4e-3 rtol=4e-3
                @test vecs[1, 1] ≈ vecs[2, 1]
                @test vecs[1, 2] ≈ -vecs[2, 2]

                radial_dc = (scale_radial_dc == 0 ? nothing :
                    IC.Function1D(x->scale_radial_dc))
                radial_rf = (scale_radial_rf == 0 ? nothing :
                    IC.Function1D(x->scale_radial_rf))

                freqs, vecs = IC.radial_modes(ions, pos, radial_dc, radial_rf)
                @test freqs[2]^2 - freqs[1] * abs(freqs[1]) ≈ real_x2 * 2 / mass atol=5e-3 rtol=5e-3
                @test freqs[2] ≈ sqrt(scale_radial_dc + scale_radial_rf / mass^2) / sqrt(mass)
                @test vecs[1, 1] ≈ -vecs[2, 1]
                @test vecs[1, 2] ≈ vecs[2, 2]
            end
        end
    end

    for (func_dc, func_rf) in ((func_dc1, func_rf1), (func_dc2, func_rf2),
                               (func_dc3, func_rf3))
        for mass in (1, 2)
            check_harmonic(mass, func_dc, func_rf, 1, 0, 1, 0)
            check_harmonic(mass, func_dc, func_rf, 0, 1, 0, 1)
            check_harmonic(mass, func_dc, func_rf, 1, 1, 1, 1)
        end
    end
end
