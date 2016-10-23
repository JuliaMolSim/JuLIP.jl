using JuLIP: stress
using JuLIP.Potentials, JuLIP.ASE

println("-------------------------------------------------")
println("   Variable Cell Test")
println("-------------------------------------------------")
calc = lennardjones(r0=JuLIP.ASE.rnn("Al"))
at = bulk("Al") * 2      # cubic=true,
set_pbc!(at, true)
set_calculator!(at, calc)

# perturb the cell shape
set_defm!(at, defm(at) + 0.1*rand(JMatF), updatepositions=true)
# check that under this deformation the atom positions are still
# in a homogeneous lattice (i.e. they are deformed correctly with the cell)
@test maxnorm(forces(at)) < 1e-12
# now perturb the atom positions as well
rattle!(at, 0.1)

# set the constraint >>> this means the deformed cell defines F0 and X0
set_constraint!(at, ExpVariableCell(at))

# check that energy, forces, stress, dofs, gradient evaluate at all
print("check that energy, forces, stress, dofs, gradient evaluate ... ")
energy(at)
forces(at)
gradient(at)
stress(at)
println("ok.")
@test true

print("Check that setting and getting dofs is consistent ... ")
x = dofs(at)
@assert length(x) == 3 * length(at) + 6
y = x + 0.01 * rand(size(x))
set_dofs!(at, y)
z = dofs(at)
@test vecnorm(y - z, Inf) < 1e-12
println("ok")

# perform the finite-difference test
set_dofs!(at, x)
JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=0.1)


# println("-------------------------------------------------")
# println("Test optimisation with ExpVariableCell")
# # # start with a clean `at`
# at = Atoms("Si") * 2   # cubic=true,
# set_pbc!(at, true)
# set_calculator!(at, StillingerWeber())
# set_constraint!(at, ExpVariableCell(at))
# #
# println("For the initial state, stress is far from 0:")
# @show vecnorm(stress(at), Inf)
# @show vecnorm(dofs(at), Inf)
# # JuLIP.Solve.minimise!(at, verbose=2)
# dt = 1e-4
# for n = 1:100
#    x = dofs(at)
#    ∇E = gradient(at)
#    @printf(" %3d  |  %4.2e \n", n, vecnorm(∇E, Inf))
#    x -= dt * ∇E
#    set_dofs!(at, x)
# end
# println("After optimisation, stress is 0:")
# @show vecnorm(stress(at), Inf)
# # @test vecnorm(stress(at), Inf) < 1e-4




# println("-------------------------------------------------")
# println("And now with pressure . . .")
# println("-------------------------------------------------")
# set_constraint!(at, ExpVariableCell(at, pressure=10.0123))
# JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=0.1)
# at = Atoms("Al", pbc=(true,true,true)) * 2   # cubic=true,
# set_calculator!(at, calc)
# set_constraint!(at, VariableCell(at, pressure=0.01))
# JuLIP.Solve.minimise!(at)
# @show vecnorm(stress(at), Inf)
# @test vecnorm(gradient(at), Inf) < 1e-4
