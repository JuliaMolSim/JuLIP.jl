
using JuLIP
using Base.Test
using JuLIP.Testing


verbose=true

julip_tests = [
   # "testaux.jl";
   # "testase.jl";
   # "testanalyticpotential.jl";
   # "testpotentials.jl";
   # "testsolve.jl";
]

println("Starting JuLIP Tests")
println("=====================")

for test in julip_tests
   include(test)
end

using JuLIP.Potentials

println("-------------------------------------------------")
println("   Variable Cell Test")
println("-------------------------------------------------")
calc = lennardjones(r0=JuLIP.ASE.rnn("Al"))
at = Atoms("Al", cubic=true, pbc=(true,true,true))
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at))
rattle!(at, 0.1)
energy(at)
forces(at)
x = dofs(at)
set_dofs!(at, x)
gradient(at, x)
JuLIP.Testing.fdtest(calc, at, verbose=true)
