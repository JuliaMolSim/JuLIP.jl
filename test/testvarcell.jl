using JuLIP.Potentials

println("-------------------------------------------------")
println("   Variable Cell Test")
println("-------------------------------------------------")
calc = lennardjones(r0=JuLIP.ASE.rnn("Al"))
at = Atoms("Al", pbc=(true,true,true)) * 2      # cubic=true,
set_calculator!(at, calc)

# perturb the cell shape
set_defm!(at, defm(at) + 0.1*rand(JMatF), updatepositions=true)
# check that under this deformation the atom positions are still
# in a homogeneous lattice (i.e. they are deformed correctly with the cell)
@test maxnorm(forces(at)) < 1e-12
# now perturb the atom positions as well
rattle!(at, 0.1)

# set the constraint >>> this means the deformed cell defines F0 and X0
set_constraint!(at, VariableCell(at))

print("check that energy, forces, virial, dofs, gradient evaluate ... ")
energy(at)
forces(at)
gradient(at)
virial(at)
println("ok.")
@test true

print("Check that setting and getting dofs is consistent ... ")
x = dofs(at)
@assert length(x) == 3 * length(at) + 9
y = x + 0.01 * rand(size(x))
set_dofs!(at, y)
z = dofs(at)
@test vecnorm(y - z, Inf) < 1e-14
println("ok")

# now perform the finite-difference test
set_dofs!(at, x)
JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=0.1)



println("-------------------------------------------------")
println("Test optimisation with VariableCell")
# start with a clean `at`
at = Atoms("Al", pbc=(true,true,true)) * 2   # cubic=true,
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at))

println("For the initial state, stress/virial is far from 0:")
@show vecnorm(virial(at), Inf)
JuLIP.Solve.minimise!(at)
println("After optimisation, stress/virial should be 0:")
@show vecnorm(virial(at), Inf)
@test vecnorm(virial(at), Inf) < 1e-4


println("-------------------------------------------------")
println("And now with pressure . . .")
println("-------------------------------------------------")
set_constraint!(at, VariableCell(at, pressure=10.0123))
JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=0.1)
at = Atoms("Al", pbc=(true,true,true)) * 2   # cubic=true,
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at, pressure=0.01))
JuLIP.Solve.minimise!(at)
@show vecnorm(virial(at), Inf)
@test vecnorm(gradient(at), Inf) < 1e-4
