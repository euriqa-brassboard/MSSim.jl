#!/usr/bin/julia

using MSSim: Utils as U
using BenchmarkTools
using PyPlot
using NaCsPlot

function compute_diffs(_f)
    function f(x)
        s, c = sincos(x)
        return _f(x, s, c)
    end
    x = range(0, 4.5, 30001)[2:end]
    v64 = f.(x)
    v32 = f.(Float32.(x))
    vbig = f.(big.(x))

    d64 = abs.(Float64.(v64 .- vbig))
    d32 = abs.(Float64.(v32 .- vbig))

    display(@benchmark $f($(0.1)))

    return x, d64, d32
end

const sin_c1 = U.TrigRatio{true,1,(),(1,),()}()

const sin_c1_x, sin_c1_d64, sin_c1_d32 = compute_diffs(U.sin_c1)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_c1_x, sin_c1_d64)
ylim([0, 1.5e-16])
ylabel("Error")
title("sin_c1 64")
grid()

subplot(1, 2, 2)
plot(sin_c1_x, sin_c1_d32)
ylim([0, 0.9e-7])
ylabel("Error")
title("sin_c1 32")
grid()

const sin_c2 = U.TrigRatio{true,2,(),(1,),(-1,)}()

const sin_c2_x, sin_c2_d64, sin_c2_d32 = compute_diffs(U.sin_c2)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_c2_x, sin_c2_d64)
ylim([0, 3e-16])
ylabel("Error")
title("sin_c2 64")
grid()

subplot(1, 2, 2)
plot(sin_c2_x, sin_c2_d32)
ylim([0, 2e-7])
ylabel("Error")
title("sin_c2 32")
grid()

const sin_c3 = U.TrigRatio{true,3,(),(-2,1),(2,)}()

const sin_c3_x, sin_c3_d64, sin_c3_d32 = compute_diffs(U.sin_c3)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_c3_x, sin_c3_d64)
ylim([0, 1e-15])
ylabel("Error")
title("sin_c3 64")
grid()

subplot(1, 2, 2)
plot(sin_c3_x, sin_c3_d32)
ylim([0, 4e-7])
ylabel("Error")
title("sin_c3 32")
grid()

const cos_f1 = U.TrigRatio{false,2,(1,),(),(-1,)}()

const cos_f1_x, cos_f1_d64, cos_f1_d32 = compute_diffs(U.cos_f1)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(cos_f1_x, cos_f1_d64)
ylim([0, 3e-16])
ylabel("Error")
title("cos_f1 64")
grid()

subplot(1, 2, 2)
plot(cos_f1_x, cos_f1_d32)
ylim([0, 2e-7])
ylabel("Error")
title("cos_f1 32")
grid()

const sin_f1 = U.TrigRatio{true,2,(1,),(-1,),()}()

const sin_f1_x, sin_f1_d64, sin_f1_d32 = compute_diffs(U.sin_f1)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_f1_x, sin_f1_d64)
ylim([0, 3e-16])
ylabel("Error")
title("sin_f1 64")
grid()

subplot(1, 2, 2)
plot(sin_f1_x, sin_f1_d32)
ylim([0, 2e-7])
ylabel("Error")
title("sin_f1 32")
grid()

const cos_f2 = U.TrigRatio{false,3,(2,),(-1,),(-2,)}()

const cos_f2_x, cos_f2_d64, cos_f2_d32 = compute_diffs(U.cos_f2)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(cos_f2_x, cos_f2_d64)
ylim([0, 1e-15])
ylabel("Error")
title("cos_f2 64")
grid()

subplot(1, 2, 2)
plot(cos_f2_x, cos_f2_d32)
ylim([0, 4e-7])
ylabel("Error")
title("cos_f2 32")
grid()

const sin_f2 = U.TrigRatio{true,3,(-1,),(2,),(-1,)}()

const sin_f2_x, sin_f2_d64, sin_f2_d32 = compute_diffs(U.sin_f2)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_f2_x, sin_f2_d64)
ylim([0, 1e-15])
ylabel("Error")
title("sin_f2 64")
grid()

subplot(1, 2, 2)
plot(sin_f2_x, sin_f2_d32)
ylim([0, 4e-7])
ylabel("Error")
title("sin_f2 32")
grid()

const cos_f3 = U.TrigRatio{false,4,(1,1//2),(-1,),(-1,)}()

const cos_f3_x, cos_f3_d64, cos_f3_d32 = compute_diffs(U.cos_f3)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(cos_f3_x, cos_f3_d64)
ylim([0, 1e-15])
ylabel("Error")
title("cos_f3 64")
grid()

subplot(1, 2, 2)
plot(cos_f3_x, cos_f3_d32)
ylim([0, 4e-7])
ylabel("Error")
title("cos_f3 32")
grid()

const sin_f3 = U.TrigRatio{true,4,(0,1//3),(-1,),(1,)}()

const sin_f3_x, sin_f3_d64, sin_f3_d32 = compute_diffs(U.sin_f3)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_f3_x, sin_f3_d64)
ylim([0, 1e-15])
ylabel("Error")
title("sin_f3 64")
grid()

subplot(1, 2, 2)
plot(sin_f3_x, sin_f3_d32)
ylim([0, 4e-7])
ylabel("Error")
title("sin_f3 32")
grid()

const sin_f4 = U.TrigRatio{true,5,(0,-1//3),(4,-1),(-4,)}()

const sin_f4_x, sin_f4_d64, sin_f4_d32 = compute_diffs(U.sin_f4)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_f4_x, sin_f4_d64)
ylim([0, 1e-15])
ylabel("Error")
title("sin_f4 64")
grid()

subplot(1, 2, 2)
plot(sin_f4_x, sin_f4_d32)
ylim([0, 4e-7])
ylabel("Error")
title("sin_f4 32")
grid()

const sin_f5 = U.TrigRatio{true,6,(0,-2//3),(20,-7),(-20,1)}()

const sin_f5_x, sin_f5_d64, sin_f5_d32 = compute_diffs(U.sin_f5)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_f5_x, sin_f5_d64)
ylim([0, 1e-15])
ylabel("Error")
title("sin_f5 64")
grid()

subplot(1, 2, 2)
plot(sin_f5_x, sin_f5_d32)
ylim([0, 4e-7])
ylabel("Error")
title("sin_f5 32")
grid()

NaCsPlot.maybe_show()
