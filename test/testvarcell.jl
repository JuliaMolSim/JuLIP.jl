
using JuLIP
using JuLIP.Potentials
using LinearAlgebra: det 

println("-------------------------------------------------")
println("   Variable Cell Test")
println("-------------------------------------------------")
calc = lennardjones(r0=rnn(:Al))
at = set_pbc!(bulk(:Al) * 2, true)
set_calculator!(at, calc)

println("Check atoms deform correctly with the cell")
# perturb the cell shape
set_defm!(at, defm(at) + 0.1*rand(JMatF), updatepositions=true)
# check that under this deformation the atom positions are still
# in a homogeneous lattice (i.e. they are deformed correctly with the cell)
@test maxnorm(forces(at)) < 1e-12
# now perturb the atom positions as well
rattle!(at, 0.1)

# set the constraint >>> this means the deformed cell defines F0 and X0
set_constraint!(at, VariableCell(at))

print("check that energy, forces, virial, stress, dofs, gradient evaluate ... ")
energy(at)
forces(at)
gradient(at)
# JuLIP.hessian_pos(at)
# JuLIP.hessian(at)     # this has no implementation for variable cells yet
virial(at)
@test stress(at) == - virial(at) / det(defm(at))
println("ok.")

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
@test JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=0.1)

println("Check virial for SW potential")
si = bulk(:Si, cubic=true) * 2
rattle!(si, 0.1)
set_defm!(si, defm(si) + 0.03 * rand(JMatF))
sw = StillingerWeber()
set_constraint!(si, VariableCell(si))
@test JuLIP.Testing.fdtest(calc, si, verbose=true, rattle=0.1)


println("-------------------------------------------------")
println("Test optimisation with VariableCell")
# start with a clean `at`
at = bulk(:Al) * 2   # cubic=true,
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at))

println("For the initial state, stress/virial is far from 0:")
@show vecnorm(virial(at), Inf)
JuLIP.Solve.minimise!(at, verbose=2)
println("After optimisation, stress/virial should be 0:")
@show vecnorm(virial(at), Inf)
@test vecnorm(virial(at), Inf) < 1e-4


println("-------------------------------------------------")
println("And now with pressure . . .")
set_constraint!(at, VariableCell(at, pressure=10.0123))
JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=0.1)
at = bulk(:Al) * 2
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at, pressure=0.01))
JuLIP.Solve.minimise!(at, verbose = 2)
@show vecnorm(virial(at), Inf)
@show vecnorm(gradient(at), Inf)
@test vecnorm(gradient(at), Inf) < 1e-4
println("note it is correct that virial is O(1) since we applied pressure")
