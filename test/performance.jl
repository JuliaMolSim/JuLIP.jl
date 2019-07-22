using BenchmarkTools
using JuLIP
using JuLIP: AbstractAtoms, AbstractCalculator

function perfbm(id::AbstractString, at::AbstractAtoms, calc::AbstractCalculator;
                e = true, elist = true, f = true, flist = true)
   println("--------------------------------------------------------------------------")
   println(id)
   if e
      print("Energy Assembly (without nlist): ")
      @btime energy($calc, $at)
   end
   if elist
      print("Energy Assembly (with nlist):    ")
      @btime energy($calc,  rattle!($at, 0.001))
   end
   if f
      print("Force Assembly (without nlist):  ")
      @btime forces($calc, $at)
   end
   if flist
      print("Force Assembly (with nlist):     ")
      @btime forces($calc,  rattle!($at, 0.001))
   end
end

println()
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("   JuLIP Performance Regression Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println()

perfbm("LENNARD-JONES",
         bulk(:Al, cubic=true) * (10,10,8),
         lennardjones(r0=rnn(:Al)) )

data = joinpath(dirname(@__FILE__), "..", "data") * "/"
perfbm("EAM (Splines)",
         bulk(:Fe, cubic=true) * (12,12,8),
         EAM(data * "pfe.plt", data * "ffe.plt", data * "F_fe.plt") )

perfbm("STILLINGER-WEBER",
         bulk(:Si, cubic=true) * (12,15,12),
         StillingerWeber())

##
using BenchmarkTools
using JuLIP
using JuLIP: AbstractAtoms, AbstractCalculator

at = bulk(:Al, cubic=true) * (10,10,8)
calc = lennardjones(r0=rnn(:Al))
tmp = JuLIP.alloc_temp(calc, at)
JuLIP.energy!(tmp, calc, at)


# Performance prior to restructuring
# TODO: fix performance of PairPotentials
# --------------------------------------------------------------------------
# LENNARD-JONES
# Energy Assembly (without nlist):   66.695 ms (41653 allocations: 47.65 MiB)
# Energy Assembly (with nlist):      68.778 ms (41675 allocations: 41.52 MiB)
# Force Assembly (without nlist):    73.158 ms (41657 allocations: 43.40 MiB)
# Force Assembly (with nlist):       72.561 ms (41677 allocations: 41.59 MiB)
# --------------------------------------------------------------------------
# EAM (Splines)
# Energy Assembly (without nlist):   69.074 ms (71535 allocations: 43.47 MiB)
# Energy Assembly (with nlist):      74.438 ms (71558 allocations: 44.49 MiB)
# Force Assembly (without nlist):    83.091 ms (69237 allocations: 44.23 MiB)
# Force Assembly (with nlist):       87.327 ms (69257 allocations: 44.51 MiB)
# --------------------------------------------------------------------------
# STILLINGER-WEBER
# Energy Assembly (without nlist):   78.205 ms (402680 allocations: 65.56 MiB)
# Energy Assembly (with nlist):      86.708 ms (410401 allocations: 69.01 MiB)
# Force Assembly (without nlist):    87.153 ms (411005 allocations: 67.34 MiB)
# Force Assembly (with nlist):       91.166 ms (410672 allocations: 69.42 MiB)
