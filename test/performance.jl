using BenchmarkTools
using JuLIP

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
         bulk("Al", cubic=true) * (10,10,8),
         lennardjones(r0=rnn("Al")) )

data = joinpath(dirname(@__FILE__), "..", "data") * "/"
perfbm("EAM (Splines)",
         bulk("Fe", cubic=true) * (12,12,8),
         EAM(data * "pfe.plt", data * "ffe.plt", data * "F_fe.plt") )

perfbm("STILLINGER-WEBER",
         bulk("Si", cubic=true) * (12,15,12),
         StillingerWeber())
