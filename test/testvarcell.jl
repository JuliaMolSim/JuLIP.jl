
using JuLIP.Potentials

println("-------------------------------------------------")
println("   Variable Cell Test")
println("-------------------------------------------------")
calc = lennardjones(r0=JuLIP.ASE.rnn("Al"))
at = Atoms("Al", pbc=(true,true,true)) * 2   # cubic=true,
set_calculator!(at, calc)

# mess with the reference cell to make it non-symmetric
F = JMat([1.0 0.2 0; 0 1 0; 0 0 1]) * defm(at)
set_defm!(at, F, updatepositions=true)
@test maxnorm(forces(at)) < 1e-12
rattle!(at, 0.1)

# set the constraint >>> this means the deformed cell defines F0 and X0
set_constraint!(at, VariableCell(at))

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
y = x + 0.2 * rand(x)
set_dofs!(at, y)
z = dofs(at)
@test vecnorm(y - x, Inf) < 1e-12
println("ok")

JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=true)
