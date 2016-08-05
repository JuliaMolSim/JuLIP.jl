
using JuLIP
using JuLIP.Potentials
using JuLIP.Testing

pairpotentials = [
   LennardJonesPotential();
   MorsePotential();
   SWCutoff(1.0, 3.0) * LennardJonesPotential();
   SplineCutoff(2.0, 3.0) * LennardJonesPotential();
]

println("============================================")
println("  Testing pair potential implementations ")
println("============================================")
r = linspace(0.8, 4.0, 100)
for pp in pairpotentials
   println("--------------------------------")
   println(pp)
   println("--------------------------------")
   fdtest(pp, r, verbose=verbose)
end

# =============================================================

calculators = Any[]

# [1] basic lennard-jones calculator test
push!(calculators, (  LennardJonesCalculator(r0=JuLIP.ASE.rnn("Al")),
         Atoms("Al", cubic=true, repeatcell=(3,3,2), pbc=(true,false,false)) ) )

# # [2] ASE's EMT calculator
# emt = JuLIP.ASE.EMTCalculator()
# at = Atoms("Cu", cubic=true, repeatcell=(2,2,2); pbc=(true,false,false))
# set_calculator!(at, emt)
# push!(calculators, (emt, at))


println("============================================")
println("  Testing calculator implementations ")
println("============================================")
for (calc, at) in calculators
   println("--------------------------------")
   println(typeof(calc))
   @show length(at)
   println("--------------------------------")
   fdtest(calc, at, verbose=true)
end
