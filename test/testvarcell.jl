using JuLIP.Potentials

println("-------------------------------------------------")
println("   Variable Cell Test")
println("-------------------------------------------------")
calc = lennardjones(r0=JuLIP.ASE.rnn("Al"))
at = Atoms("Al", pbc=(true,true,true)) * 2   # cubic=true,
set_calculator!(at, calc)

# mess with the reference cell to make it non-symmetric
# F = JMat([1.0 0.2 0; 0 1 0; 0 0 1]) * defm(at)
F = JMat(eye(3) + 0.1 * rand(3,3)) * defm(at)
set_defm!(at, F, updatepositions=true)
@test maxnorm(forces(at)) < 1e-12
rattle!(at, 0.1)

# set the constraint >>> this means the deformed cell defines F0 and X0
set_constraint!(at, VariableCell(at))

# check that energy, forces, stress, dofs, gradient evaluate at all
# print("check that energy, forces, stress, dofs, gradient evaluate ... ")
# energy(at)
# forces(at)
# gradient(at)
# stress(at)
# println("ok.")
# @test true

print("Check that setting and getting dofs is consistent ... ")
x = dofs(at)
@assert length(x) == 3*length(at) + 9
y = x + 0.01 * rand(size(x))
set_dofs!(at, y)
z = dofs(at)
@test vecnorm(y - z, Inf) < 1e-14
println("ok")

set_dofs!(at, x)
# rattle!(at, 0.01)
# set_defm!(at, defm(at) + 0.1*rand(JMatF))
JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=0.1)




# println("-------------------------------------------------")
# println("Test optimisation with VariableCell")
# # start with a clean `at`
# at = Atoms("Al", pbc=(true,true,true)) * 2   # cubic=true,
# set_calculator!(at, calc)
# set_constraint!(at, VariableCell(at))
#
# # println("For the initial state, stress is far from 0:")
# # @show vecnorm(stress(at), Inf)
# # JuLIP.Solve.minimise!(at)
# # println("After optimisation, stress is 0:")
# # @show vecnorm(stress(at), Inf)
# # @test vecnorm(stress(at), Inf) < 1e-8



# println("-------------------------------------------------")
# println("And now with pressure ...")
# set_constraint!(at, VariableCell(at, pressure=10.0123))
# JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=true)
# at = Atoms("Al", pbc=(true,true,true)) * 2   # cubic=true,
# set_calculator!(at, calc)
# set_constraint!(at, VariableCell(at, pressure=0.0))
# JuLIP.Solve.minimise!(at)
# @show vecnorm(stress(at), Inf)
