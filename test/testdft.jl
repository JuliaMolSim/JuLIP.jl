
println("-------------------")
println(" Testing JuLIP.DFT")
println("-------------------")

using JuLIP
using JuLIP.ASE
using JuLIP.DFT

# ======================================================================

at = bulk("Si", cubic=true)
g = GPAWCalculator(xc="LDA",
                   gpts=(32, 32, 32),
                   nbands="150%",
                   kpts=(1, 1, 1),
                   occupations=FermiDirac(width=0.01),
		   convergence=Dict("energy" => 1e-7,
                                    "eigenstates" => 1e-10,
                                    "density" => 1e-7))
println(g)
set_calculator!(at, g)

E = energy(g, at)
println("E = ", E)

E_a = site_energies(g,at)
println("E_a = ", E_a)
println("sum(E_a) = ", sum(E_a))
println("E - sum(E_a) = ", E - sum(E_a))

@test (E - sum(E_a)) < 1e-3

