using BenchmarkTools
using JuLIP, JuLIP.ASE, JuLIP.Potentials

println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("   JuLIP Performance Regression Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println()
println("LENNARD-JONES")
println("==============")

calc = lennardjones(r0=JuLIP.ASE.rnn("Al"))
at = bulk("Al", cubic=true) * (20,20,2)
set_calculator!(at, calc)
set_constraint!(at, FixedCell(at))

println("Energy Assembly (without nlist)")
@btime energy($calc, $at)

println("Energy Assembly (with nlist)")
f() = energy(calc,  rattle!(at,0.001))
@btime f()
