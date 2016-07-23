using JuLIP
using JuLIP.Potentials
using JuLIP.Testing

# pairpotentials = [
#    LennardJonesPotential();
#    MorsePotential();
#    SimpleExponential();
#    SWCutoff(LennardJonesPotential(), 1.0, 3.0);
#    SplineCutoff(LennardJonesPotential(), 2.0, 3.0)
# ]
#
# println("============================================")
# println("  Testing pair potential implementations ")
# println("============================================")
# r = linspace(0.8, 4.0, 100)
# for pp in pairpotentials
#    println("--------------------------------")
#    println(typeof(pp))
#    println("--------------------------------")
#    fdtest(pp, r, verbose=verbose)
# end


calculators = [
   (  LennardJonesCalculator(r0=JuLIP.ASE.rnn("Al")),
      Atoms("Al", cubic=true, repeatcell=(3,3,2), pbc=(true,false,false))  );
]

println("============================================")
println("  Testing calculator implementations ")
println("============================================")
r = linspace(0.8, 4.0, 100)
for (calc, at) in calculators
   println("--------------------------------")
   println(typeof(calc))
   @show length(at)
   println("--------------------------------")
   fdtest(calc, at, verbose=true)
end
