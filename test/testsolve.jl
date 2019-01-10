
using JuLIP
using Base.Test

println("===================================================")
println("          TEST SOLVE ")
println("===================================================")

println("-----------------------------------------------------------------")
println("Testing `minimise!` with equilibration with LJ calculator to lattice")
println("-----------------------------------------------------------------")
calc = lennardjones(r0=rnn(:Al))
at = bulk(:Al, cubic=true) * 10
X0 = positions(at) |> mat
at = rattle!(at, 0.02)
set_calculator!(at, calc)
set_constraint!(at, FixedCell(at))
minimise!(at, precond=:id, verbose=2)
X1 = positions(at) |> mat
X0 .-= X0[:, 1]
X1 .-= X1[:, 1]
F = X1 / X0
println("check that the optimiser really converged to a lattice")
@show vecnorm(F'*F - eye(3), Inf)
@show vecnorm(F*X0 - X1, Inf)
@test vecnorm(F*X0 - X1, Inf) < 1e-4

println("-------------------------------------------------")
println("same test but large and with Exp preconditioner")
println("-------------------------------------------------")

at = bulk(:Al, cubic=true) * (20,20,2)
at = rattle!(at, 0.02)
set_calculator!(at, calc)
set_constraint!(at, FixedCell(at))
minimise!(at, precond = :exp, method = :lbfgs,
          robust_energy_difference = true, verbose=2)


println("-------------------------------------------------")
println("   Variable Cell Test")
println("-------------------------------------------------")
calc = lennardjones(r0=rnn(:Al))
at = set_pbc!(bulk(:Al, cubic=true), true)
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at))
minimise!(at, verbose = 2)


println("-------------------------------------------------")
println(" FF preconditioner for StillingerWeber ")
println("-------------------------------------------------")

at = bulk(:Si, cubic=true) * (10,10,2)
at = set_pbc!(at, true)
at = rattle!(at, 0.02)
set_calculator!(at, StillingerWeber())
set_constraint!(at, FixedCell(at))
P = FF(at, StillingerWeber())
minimise!(at, precond = P, method = :lbfgs, robust_energy_difference = true, verbose=2)


println("-------------------------------------------------")
println(" FF preconditioner for EAM ")
println("-------------------------------------------------")

at = bulk(:W, cubic=true) * (10,10,2)
at = set_pbc!(at, true)
at = rattle!(at, 0.02)
X0 = positions(at)

##
set_positions!(at, X0)
set_calculator!(at, eam_W)
set_constraint!(at, FixedCell(at))
P = FF(at, eam_W)
minimise!(at, precond = P, method = :lbfgs, robust_energy_difference = true, verbose=2)

## steepest descent
set_positions!(at, X0)
set_calculator!(at, eam_W)
set_constraint!(at, FixedCell(at))
P = FF(at, eam_W)
minimise!(at, precond = P, method = :sd, robust_energy_difference = true, verbose=2)


##
println("Optimise again with some different stabilisation options")
set_positions!(at, X0)
set_calculator!(at, eam_W)
set_constraint!(at, FixedCell(at))
P = FF(at, eam_W, stab=0.1, innerstab=0.2)
minimise!(at, precond = P, method = :lbfgs, robust_energy_difference = true, verbose=2)

##
println("for comparison now with Exp")
set_positions!(at, X0)
minimise!(at, precond = :exp, method = :lbfgs, robust_energy_difference = true, verbose=2)
