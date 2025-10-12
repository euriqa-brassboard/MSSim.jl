#!/usr/bin/julia

using Distributed

addprocs(Sys.CPU_THREADS)

pmap(["sequence_objective",
      "sym_linear",
      "segmented_sequence",
      "utils",
      "sequence",
      "pure_numeric",
      "ion_chain",
      "optimizers",
      "far_detune"]) do file
          println("Start testing $file")
          @eval module $(Symbol("Test_$(file)_mod"))
          include($(joinpath(@__DIR__, "$(file).jl")))
          end
          println("Done testing $file")
          # So that we do not try to bring the worker-only module back to the main process
          return
      end
