
println("-------------------")
println(" Testing JuLIP.DFT")
println("-------------------")

using JuLIP
using JuLIP.ASE
using JuLIP.DFT

# ======================================================================

at = bulk("Si", cubic=true)
g = GPAWCalculator()
println(g)
set_calculator!(at, g)

eden = energy_density(g, at)
E_a = site_energies(g, at)
