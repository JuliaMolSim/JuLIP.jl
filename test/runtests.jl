
using JuLIP
using Base.Test
using JuLIP.Testing


verbose=true

julip_tests = [
   "testaux.jl";
   "testase.jl";
   "testdft.jl";
   "testanalyticpotential.jl";
   "testpotentials.jl";
   "testvarcell.jl";
   # "testexpvarcell.jl";
   "testsolve.jl";
]

println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("  Starting JuLIP Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")


for test in julip_tests
   include(test)
end

# using JuLIP.Potentials, JuLIP.ASE
#
# println("Testing `site_energy` ...")
# at = bulk("Si", pbc=true, cubic=true) * 3
# sw = StillingerWeber()
# atsm = bulk("Si", pbc = true)
# println(" ... passed site_energy identity, now testing derivative ...")
# @test abs( JuLIP.Potentials.site_energy(sw, at, 1) - energy(sw, atsm) / 2 ) < 1e-10
#
# # finite-difference test
# set_constraint!(at, FixedCell(at))
# f(x) = JuLIP.Potentials.site_energy(sw, set_dofs!(at, x), 1)
# df(x) = (JuLIP.Potentials.site_energy_d(sw, set_dofs!(at, x), 1) |> mat)[:]
# @test fdtest(f, df, dofs(at); verbose=true)
