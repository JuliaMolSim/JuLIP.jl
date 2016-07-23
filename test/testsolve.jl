
using JuLIP
using JuLIP.Potentials
using JuLIP.Solve
using JuLIP.Constraints

println("===================================================")
println("          TEST SOLVE ")
println("===================================================")

println("Testing `minimise!` with equilibration with LJ calculator to lattice")
calc = LennardJonesCalculator(r0=JuLIP.ASE.rnn("Al"))
at = Atoms("Al", cubic=true, repeatcell=(3,3,3), pbc=(true,true,true))
X0 = positions(at) |> mat
at = rattle!(at, 0.02)
set_calculator!(at, calc)
set_constraint!(at, FixedCell(at));
minimise!(at)
X1 = positions(at) |> mat
X0 .-= X0[:, 1]
X1 .-= X1[:, 1]
F = X1 / X0
@assert vecnorm(F'*F - eye(3), Inf) < 1e-6
@assert vecnorm(F*X0 - X1, Inf) < 1e-6
