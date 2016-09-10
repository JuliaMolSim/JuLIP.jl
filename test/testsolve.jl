
using JuLIP
using JuLIP.Potentials
using JuLIP.Solve
using JuLIP.Constraints
import JuLIP.Preconditioners: Exp

println("===================================================")
println("          TEST SOLVE ")
println("===================================================")

println("-----------------------------------------------------------------")
println("Testing `minimise!` with equilibration with LJ calculator to lattice")
println("-----------------------------------------------------------------")
calc = lennardjones(r0=JuLIP.ASE.rnn("Al"))
at = Atoms("Al", cubic=true, repeatcell=(3,3,3), pbc=(true,true,true))
X0 = positions(at) |> mat
at = rattle!(at, 0.02)
set_calculator!(at, calc)
set_constraint!(at, FixedCell(at))
minimise!(at)
X1 = positions(at) |> mat
X0 .-= X0[:, 1]
X1 .-= X1[:, 1]
F = X1 / X0
println("check that the optimiser really converged to a lattice")
@show vecnorm(F'*F - eye(3), Inf)
@show vecnorm(F*X0 - X1, Inf)


println("-------------------------------------------------")
println("same test but large and with Exp preconditioner")
println("-------------------------------------------------")

at = Atoms("Al", repeatcell=(20,20,2), pbc=(true,true,true), cubic=true)
at = rattle!(at, 0.02)
set_calculator!(at, calc)
set_constraint!(at, FixedCell(at))
minimise!(at, precond = Exp(at))
