
using JuLIP
using Base.Test
using JuLIP.Testing


verbose=true

julip_tests = [
   # "testaux.jl";
   # "testase.jl";
   # "testanalyticpotential.jl";
   # "testpotentials.jl";
   # "testvarcell.jl";
   # "testsolve.jl";
]

println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("  Starting JuLIP Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")


for test in julip_tests
   include(test)
end


si = JuLIP.ASE.bulk("Si", cubic=true) * 10
sw = JuLIP.Potentials.StillingerWeber()
@time energy(sw, si);
@time energy(sw, si);
@time energy(sw, si);
@time forces(sw, si);
@time forces(sw, si);
@time forces(sw, si);
