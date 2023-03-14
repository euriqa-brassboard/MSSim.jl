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

const sinc_x, sinc_d64_small, sinc_d64_big, sinc_d32_small, sinc_d32_big =
    compute_diffs(MSSim.Utils._sinc_small, MSSim.Utils._sinc_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sinc_x, sinc_d64_big, label="big")
plot(sinc_x, sinc_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 1.5e-16])
ylabel("Error")
title("sinc 64")

subplot(1, 2, 2)
plot(sinc_x, sinc_d32_big, label="big")
plot(sinc_x, sinc_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 0.9e-7])
ylabel("Error")
title("sinc 32")

const cosc_x, cosc_d64_small, cosc_d64_big, cosc_d32_small, cosc_d32_big =
    compute_diffs(MSSim.Utils._cosc_small, MSSim.Utils._cosc_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(cosc_x, cosc_d64_big, label="big")
plot(cosc_x, cosc_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 3e-16])
ylabel("Error")
title("cosc 64")

subplot(1, 2, 2)
plot(cosc_x, cosc_d32_big, label="big")
plot(cosc_x, cosc_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 2e-7])
ylabel("Error")
title("cosc 32")

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

subplot(1, 2, 2)
plot(cos_f1_x, cos_f1_d32_big, label="big")
plot(cos_f1_x, cos_f1_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 2e-7])
ylabel("Error")
title("cos_f1 32")

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

subplot(1, 2, 2)
plot(sin_f1_x, sin_f1_d32_big, label="big")
plot(sin_f1_x, sin_f1_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 2e-7])
ylabel("Error")
title("sin_f1 32")

const cos_f2_x, cos_f2_d64_small, cos_f2_d64_big, cos_f2_d32_small, cos_f2_d32_big =
    compute_diffs(MSSim.Utils._cos_f2_small, MSSim.Utils._cos_f2_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(cos_f2_x, cos_f2_d64_big, label="big")
plot(cos_f2_x, cos_f2_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 3e-16])
ylabel("Error")
title("cos_f2 64")

subplot(1, 2, 2)
plot(cos_f2_x, cos_f2_d32_big, label="big")
plot(cos_f2_x, cos_f2_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 2e-7])
ylabel("Error")
title("cos_f2 32")

const sin_f2_x, sin_f2_d64_small, sin_f2_d64_big, sin_f2_d32_small, sin_f2_d32_big =
    compute_diffs(MSSim.Utils._sin_f2_small, MSSim.Utils._sin_f2_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_f2_x, sin_f2_d64_big, label="big")
plot(sin_f2_x, sin_f2_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 3e-16])
ylabel("Error")
title("sin_f2 64")

subplot(1, 2, 2)
plot(sin_f2_x, sin_f2_d32_big, label="big")
plot(sin_f2_x, sin_f2_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 2e-7])
ylabel("Error")
title("sin_f2 32")

const cos_f3_x, cos_f3_d64_small, cos_f3_d64_big, cos_f3_d32_small, cos_f3_d32_big =
    compute_diffs(MSSim.Utils._cos_f3_small, MSSim.Utils._cos_f3_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(cos_f3_x, cos_f3_d64_big, label="big")
plot(cos_f3_x, cos_f3_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 3e-16])
ylabel("Error")
title("cos_f3 64")

subplot(1, 2, 2)
plot(cos_f3_x, cos_f3_d32_big, label="big")
plot(cos_f3_x, cos_f3_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 2e-7])
ylabel("Error")
title("cos_f3 32")

const sin_f3_x, sin_f3_d64_small, sin_f3_d64_big, sin_f3_d32_small, sin_f3_d32_big =
    compute_diffs(MSSim.Utils._sin_f3_small, MSSim.Utils._sin_f3_big)

figure(figsize=[6.4 * 2, 4.8])

subplot(1, 2, 1)
plot(sin_f3_x, sin_f3_d64_big, label="big")
plot(sin_f3_x, sin_f3_d64_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 3e-16])
ylabel("Error")
title("sin_f3 64")

subplot(1, 2, 2)
plot(sin_f3_x, sin_f3_d32_big, label="big")
plot(sin_f3_x, sin_f3_d32_small, label="small")
legend(fontsize=10, ncol=2)
ylim([0, 2e-7])
ylabel("Error")
title("sin_f3 32")

NaCsPlot.maybe_show()
