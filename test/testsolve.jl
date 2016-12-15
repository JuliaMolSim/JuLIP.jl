
using JuLIP, JuLIP.ASE, JuLIP.Potentials, JuLIP.Solve, JuLIP.Constraints
using JuLIP.Preconditioners: Exp

println("===================================================")
println("          TEST SOLVE ")
println("===================================================")

println("-----------------------------------------------------------------")
println("Testing `minimise!` with equilibration with LJ calculator to lattice")
println("-----------------------------------------------------------------")
calc = lennardjones(r0=JuLIP.ASE.rnn("Al"))
at = bulk("Al", cubic=true) * 3
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

#at = bulk("Al", cubic=true) * (20,20,2)
#at = set_pbc!(at, true)
#at = rattle!(at, 0.02)
#set_calculator!(at, calc)
#set_constraint!(at, FixedCell(at))
#minimise!(at, precond = Exp(at))


println("-------------------------------------------------")
println("   Variable Cell Test")
println("-------------------------------------------------")
calc = lennardjones(r0=JuLIP.ASE.rnn("Al"))
at = set_pbc!(bulk("Al", cubic=true), true)
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at))
