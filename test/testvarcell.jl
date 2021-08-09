
using JuLIP, Test
using JuLIP.Potentials, JuLIP.Testing
using LinearAlgebra

calc = lennardjones(r0=rnn(:Al))
at = set_pbc!(bulk(:Al) * 2, true)
set_calculator!(at, calc)
println(@test maxnorm(forces(at)) < 1e-12)

h3("Check atoms deform correctly with the cell")
# perturb the cell shape
apply_defm!(at, I + 0.01*rand(JMatF))
# check that under this deformation the atom positions are still
# in a homogeneous lattice (i.e. they are deformed correctly with the cell)
println(@test maxnorm(forces(at)) < 1e-12)
# now perturb the atom positions as well
rattle!(at, 0.1)

# set the constraint >>> this means the deformed cell defines F0 and X0
variablecell!(at)

h3("check that energy, forces, virial, stress, dofs, gradient evaluate ... ")
energy(at)
forces(at)
gradient(at)
virial(at)
println(@test stress(at) ≈ - virial(at) / volume(at))

h3("Check that setting and getting dofs is consistent ... ")
x = dofs(at)
@assert length(x) == 3 * length(at) + 9
y = x + 0.01 * rand(Float64, size(x))
set_dofs!(at, y)
z = dofs(at)
println(@test norm(y - z, Inf) < 1e-14)

# now perform the finite-difference test
set_dofs!(at, x)
println(@test JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=0.1))

h3("Check virial for SW potential")
si = bulk(:Si, cubic=true) * 2
rattle!(si, 0.1)
apply_defm!(at, I + 0.01*rand(JMatF))
sw = StillingerWeber()
variablecell!(si)
println(@test JuLIP.Testing.fdtest(calc, si, verbose=true, rattle=0.1))
