#!/usr/bin/julia

using Distributed

addprocs(Sys.CPU_THREADS)

pmap(["utils.jl",
      "pure_numeric.jl",
      "sym_linear.jl",
      "segmented_sequence.jl",
      "ion_chain.jl",
      "sequence.jl",
      "optimizers.jl"]) do file
          println("Start testing $file")
          include(joinpath(@__DIR__, file))
          println("Done testing $file")
          # So that we do not try to bring the worker-only module back to the main process
          return
      end
