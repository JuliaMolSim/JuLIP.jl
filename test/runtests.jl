
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
set_constraint!(at, VariableCell(at))

# rattle!(at, 0.1)
# energy(at)
# forces(at)
# x = dofs(at)
# set_dofs!(at, x)
# gradient(at, x)
# JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=false)


# deform cell a bit
X0 = positions(at)
F = cell(at)'
dF = eye(3) + 0.1 * rand(3,3)
set_cell!(at, (dF*F)')
set_positions!(at,  [JVec(dF * x) for x in X0])
# create non-zero forces
# X = positions(at)
# X[1] += 0.1
# set_positions!(at, X)
rattle!(at, 0.1)
@show maxnorm(forces(at))
@show cell(at)'
println("...............................")

F0 = cell(at)'
F = copy(F0)'
X0 = positions(at)
∂W = (stress(at) |> Array)  / F
E0 = energy(at)
∂Wh = zeros(∂W)
for p = 3:7
   h = 0.1^p
   for i = 1:9
      U = zeros(3,3); U[i] += h
      F = Matrix(F0 + U)
      set_cell!(at, F')
      A = JMat( F * pinv(F0) )
      X = [ A * x for x in X0 ]
      set_positions!(at, X)
      ∂Wh[i] = (energy(at) - E0) / h
   end
   println(p, " -> ", vecnorm(∂W - ∂Wh, Inf))
end

@show ∂W
@show ∂Wh
