
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
at = Atoms("Al", pbc=(true,true,true)) * 2   # cubic=true,
set_calculator!(at, calc)

# mess with the cell
F = JMat([1.0 0.2 0; 0 1 0; 0 0 1]) * defm(at)
set_defm!(at, F, updatepositions=true)
@test maxnorm(forces(at)) < 1e-12
rattle!(at, 0.1)

# set the constraint >>> this means the deformed cell defined F0
set_constraint!(at, VariableCell(at))

# check that energy, forces, stress, dofs, gradient evaluate at all
print("check that energy, forces, stress, dofs, gradient evaluate ... ")
energy(at)
forces(at)
gradient(at)
stress(at)
println("ok.")

print("Check that setting and getting dofs is consistent ... ")
x = dofs(at)
y = x + 0.1 * rand(x)
set_dofs!(at, y)
z = dofs(at)
@test vecnorm(y - x, Inf) < 1e-12
println("ok")

JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=true)


# quit()
#
# # deform cell a bit
# # create non-zero forces
# # X = positions(at)
# # X[1] += 0.1
# # set_positions!(at, X)
# rattle!(at, 0.1)
# @show cell(at)'
# println("...............................")
#
# F0 = cell(at)'
# F = copy(F0)'
# X0 = positions(at)
# ∂W = (stress(at) |> Array)  / F
# E0 = energy(at)
# ∂Wh = zeros(∂W)
# for p = 3:7
#    h = 0.1^p
#    for i = 1:9
#       U = zeros(3,3); U[i] += h
#       F = Matrix(F0 + U)
#       set_cell!(at, F')
#       A = JMat( F * pinv(F0) )
#       X = [ A * x for x in X0 ]
#       set_positions!(at, X)
#       ∂Wh[i] = (energy(at) - E0) / h
#    end
#    println(p, " -> ", vecnorm(∂W - ∂Wh, Inf))
# end
#
# @show ∂W
# @show ∂Wh
