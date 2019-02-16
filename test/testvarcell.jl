
using JuLIP
using JuLIP.Potentials
using LinearAlgebra

calc = lennardjones(r0=rnn(:Al))
at = set_pbc!(bulk(:Al) * 2, true)
set_calculator!(at, calc)

h3("Check atoms deform correctly with the cell")
# perturb the cell shape
set_defm!(at, defm(at) + 0.1*rand(JMatF), updatepositions=true)
# check that under this deformation the atom positions are still
# in a homogeneous lattice (i.e. they are deformed correctly with the cell)
println(@test maxnorm(forces(at)) < 1e-12)
# now perturb the atom positions as well
rattle!(at, 0.1)

# set the constraint >>> this means the deformed cell defines F0 and X0
set_constraint!(at, VariableCell(at))

h3("check that energy, forces, virial, stress, dofs, gradient evaluate ... ")
energy(at)
forces(at)
JuLIP.gradient(at)
# JuLIP.hessian_pos(at)
# JuLIP.hessian(at)     # this has no implementation for variable cells yet
virial(at)
println(@test stress(at) == - virial(at) / det(defm(at)))
println("ok.")

h3("Check that setting and getting dofs is consistent ... ")
x = dofs(at)
@assert length(x) == 3 * length(at) + 9
y = x + 0.01 * rand(Float64, size(x))
set_dofs!(at, y)
z = dofs(at)
println(@test norm(y - z, Inf) < 1e-14)
println("ok")

# now perform the finite-difference test
set_dofs!(at, x)
println(@test JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=0.1))

h3("Check virial for SW potential")
si = bulk(:Si, cubic=true) * 2
rattle!(si, 0.1)
set_defm!(si, defm(si) + 0.03 * rand(JMatF))
sw = StillingerWeber()
set_constraint!(si, VariableCell(si))
println(@test JuLIP.Testing.fdtest(calc, si, verbose=true, rattle=0.1))
