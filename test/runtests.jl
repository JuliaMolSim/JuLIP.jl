
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

function polar(F::Matrix)
   @assert size(F,1) == size(F,2)
   Q1, D, Q2 = svd(F)
   return Q1 * Q2', Q2 * diagm(D) * Q2'
end

# deform cell a bit
X = positions(at)
F = cell(at)
dF = eye(3,3) + rand(3,3) * 0.1
# dF = [1.0 0.1 0.0; 0.1 1.0 0.0; 0.0 0.0 1.0]
set_cell!(at, dF * F * dF')
set_positions!(at,  [JVec(dF*F*dF' * (F \ x)) for x in X] )
# create non-zero forces
# X = positions(at)
# X[1] += 0.1
# set_positions!(at, X)
rattle!(at, 0.1)
@show maxnorm(forces(at))
@show cell(at)

F0 = cell(at)
F = copy(F0)
X0 = positions(at)
∂W = (stress(at) |> Array)  / F'
E0 = energy(at)
∂Wh = zeros(∂W)
for p = 3:7
   h = 0.1^p
   for i = 1:9
      U = zeros(3,3); U[i] += h
      F = Matrix(F0 + U)
      Q, F = polar(F)
      A = JMat( F * pinv(F0) )
      X = [ A * x for x in X0 ]
      set_cell!(at, F)
      set_positions!(at, X)
      ∂Wh[i] = (energy(at) - E0) / h
   end
   println(p, " -> ", vecnorm(∂W - ∂Wh, Inf))
end

@show ∂W
@show ∂Wh


# # check why rotation changes energy
#
# E0 = energy(at)
#
# # generate a rotation matrix
# U = [0 1.0 0; -1.0 0 0; 0 0 0]
# h = 0.1
# Q = expm(h * U)
# @assert abs(det(Q) - 1.0) < 1e-10
# @assert vecnorm(Q'*Q - eye(3)) < 1e-10
#
# # deform cell
# F0 = cell(at)
# X0 = positions(at)
# F = Q * F0
# X = [ JMat(Q) * x for x in X0 ]
# set_cell!(at, F)
# set_positions!(at, X)
#
# # compute energy after rotation
# Eh = energy(at)
# @show E0, Eh
# # output
# #    (E0,Eh) = (-126.23143471478076,-112.77456617599405)
