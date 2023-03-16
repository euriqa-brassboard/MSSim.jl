#!/usr/bin/julia

using MSSim
using BenchmarkTools
using PyPlot
using NaCsPlot

function compute_diffs(f_small, _f_big)
    function f_big(x)
        s, c = sincos(x)
        return _f_big(x, s, c)
    end
    x = range(0, 1, 10001)[2:end]
    v64_small = f_small.(x)
    v64_big = f_big.(x)
    v32_small = f_small.(Float32.(x))
    v32_big = f_big.(Float32.(x))
    vbig = f_big.(big.(x))

    d64_small = abs.(Float64.(v64_small .- vbig))
    d64_big = abs.(Float64.(v64_big .- vbig))
    d32_small = abs.(Float64.(v32_small .- vbig))
    d32_big = abs.(Float64.(v32_big .- vbig))

    display(@benchmark $f_small($(0.1)))
    display(@benchmark $f_big($(0.1)))

    return x, d64_small, d64_big, d32_small, d32_big
end

const sin_c1_x, sin_c1_d64_small, sin_c1_d64_big, sin_c1_d32_small, sin_c1_d32_big =
    compute_diffs(MSSim.Utils._sin_c1_small, MSSim.Utils._sin_c1_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_c1_x, sin_c1_d64_big, label="big")
plot(sin_c1_x, sin_c1_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 1.5e-16])
ylabel("Error")
title("sin_c1 64")
grid()

subplot(1, 2, 2)
plot(sin_c1_x, sin_c1_d32_big, label="big")
plot(sin_c1_x, sin_c1_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 0.9e-7])
ylabel("Error")
title("sin_c1 32")
grid()

const sin_c2_x, sin_c2_d64_small, sin_c2_d64_big, sin_c2_d32_small, sin_c2_d32_big =
    compute_diffs(MSSim.Utils._sin_c2_small, MSSim.Utils._sin_c2_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_c2_x, sin_c2_d64_big, label="big")
plot(sin_c2_x, sin_c2_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 3e-16])
ylabel("Error")
title("sin_c2 64")
grid()

subplot(1, 2, 2)
plot(sin_c2_x, sin_c2_d32_big, label="big")
plot(sin_c2_x, sin_c2_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 2e-7])
ylabel("Error")
title("sin_c2 32")
grid()

const cos_f1_x, cos_f1_d64_small, cos_f1_d64_big, cos_f1_d32_small, cos_f1_d32_big =
    compute_diffs(MSSim.Utils._cos_f1_small, MSSim.Utils._cos_f1_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(cos_f1_x, cos_f1_d64_big, label="big")
plot(cos_f1_x, cos_f1_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 3e-16])
ylabel("Error")
title("cos_f1 64")
grid()

subplot(1, 2, 2)
plot(cos_f1_x, cos_f1_d32_big, label="big")
plot(cos_f1_x, cos_f1_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 2e-7])
ylabel("Error")
title("cos_f1 32")
grid()

const sin_f1_x, sin_f1_d64_small, sin_f1_d64_big, sin_f1_d32_small, sin_f1_d32_big =
    compute_diffs(MSSim.Utils._sin_f1_small, MSSim.Utils._sin_f1_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_f1_x, sin_f1_d64_big, label="big")
plot(sin_f1_x, sin_f1_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 3e-16])
ylabel("Error")
title("sin_f1 64")
grid()

subplot(1, 2, 2)
plot(sin_f1_x, sin_f1_d32_big, label="big")
plot(sin_f1_x, sin_f1_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 2e-7])
ylabel("Error")
title("sin_f1 32")
grid()

const cos_f2_x, cos_f2_d64_small, cos_f2_d64_big, cos_f2_d32_small, cos_f2_d32_big =
    compute_diffs(MSSim.Utils._cos_f2_small, MSSim.Utils._cos_f2_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(cos_f2_x, cos_f2_d64_big, label="big")
plot(cos_f2_x, cos_f2_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 1e-15])
ylabel("Error")
title("cos_f2 64")
grid()

subplot(1, 2, 2)
plot(cos_f2_x, cos_f2_d32_big, label="big")
plot(cos_f2_x, cos_f2_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 4e-7])
ylabel("Error")
title("cos_f2 32")
grid()

const sin_f2_x, sin_f2_d64_small, sin_f2_d64_big, sin_f2_d32_small, sin_f2_d32_big =
    compute_diffs(MSSim.Utils._sin_f2_small, MSSim.Utils._sin_f2_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_f2_x, sin_f2_d64_big, label="big")
plot(sin_f2_x, sin_f2_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 1e-15])
ylabel("Error")
title("sin_f2 64")
grid()

subplot(1, 2, 2)
plot(sin_f2_x, sin_f2_d32_big, label="big")
plot(sin_f2_x, sin_f2_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 4e-7])
ylabel("Error")
title("sin_f2 32")
grid()

const cos_f3_x, cos_f3_d64_small, cos_f3_d64_big, cos_f3_d32_small, cos_f3_d32_big =
    compute_diffs(MSSim.Utils._cos_f3_small, MSSim.Utils._cos_f3_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(cos_f3_x, cos_f3_d64_big, label="big")
plot(cos_f3_x, cos_f3_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 1e-15])
ylabel("Error")
title("cos_f3 64")
grid()

subplot(1, 2, 2)
plot(cos_f3_x, cos_f3_d32_big, label="big")
plot(cos_f3_x, cos_f3_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 4e-7])
ylabel("Error")
title("cos_f3 32")
grid()

const sin_f3_x, sin_f3_d64_small, sin_f3_d64_big, sin_f3_d32_small, sin_f3_d32_big =
    compute_diffs(MSSim.Utils._sin_f3_small, MSSim.Utils._sin_f3_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_f3_x, sin_f3_d64_big, label="big")
plot(sin_f3_x, sin_f3_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 1e-15])
ylabel("Error")
title("sin_f3 64")
grid()

subplot(1, 2, 2)
plot(sin_f3_x, sin_f3_d32_big, label="big")
plot(sin_f3_x, sin_f3_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 4e-7])
ylabel("Error")
title("sin_f3 32")
grid()

NaCsPlot.maybe_show()
