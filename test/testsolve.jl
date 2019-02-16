
using JuLIP
using Test
using LinearAlgebra

h2("Testing `minimise!` with equilibration with LJ calculator to lattice")
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
@show norm(F'*F - I, Inf)
@show norm(F*X0 - X1, Inf)
@test norm(F*X0 - X1, Inf) < 1e-4

h2("same test but large and with Exp preconditioner")
at = bulk(:Al, cubic=true) * (20,20,2)
at = rattle!(at, 0.02)
set_calculator!(at, calc)
set_constraint!(at, FixedCell(at))
minimise!(at, precond = :exp, method = :lbfgs,
          robust_energy_difference = true, verbose=2)


h2("Variable Cell Test")
calc = lennardjones(r0=rnn(:Al))
at = set_pbc!(bulk(:Al, cubic=true), true)
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at))
minimise!(at, verbose = 2)


h2("FF preconditioner for StillingerWeber")
at = bulk(:Si, cubic=true) * (10,10,2)
at = set_pbc!(at, true)
at = rattle!(at, 0.02)
set_calculator!(at, StillingerWeber())
set_constraint!(at, FixedCell(at))
P = FF(at, StillingerWeber())
minimise!(at, precond = P, method = :lbfgs, robust_energy_difference = true, verbose=2)


h2("FF preconditioner for EAM")
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
h2("Optimise again with some different stabilisation options")
set_positions!(at, X0)
set_calculator!(at, eam_W)
set_constraint!(at, FixedCell(at))
P = FF(at, eam_W, stab=0.1, innerstab=0.2)
minimise!(at, precond = P, method = :lbfgs, robust_energy_difference = true, verbose=2)

##
h2("for comparison now with Exp")
set_positions!(at, X0)
minimise!(at, precond = :exp, method = :lbfgs, robust_energy_difference = true, verbose=2)


h2("Test optimisation with VariableCell")
# start with a clean `at`
at = bulk(:Al) * 2   # cubic=true,
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at))

h2("For the initial state, stress/virial is far from 0:")
@show norm(virial(at), Inf)
JuLIP.Solve.minimise!(at, verbose=2)
println("After optimisation, stress/virial should be 0:")
@show norm(virial(at), Inf)
@test norm(virial(at), Inf) < 1e-4


h2("And now with pressure . . .")
set_constraint!(at, VariableCell(at, pressure=10.0123))
JuLIP.Testing.fdtest(calc, at, verbose=true, rattle=0.1)
at = bulk(:Al) * 2
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at, pressure=0.01))
JuLIP.Solve.minimise!(at, verbose = 2)
@show norm(virial(at), Inf)
@show norm(JuLIP.gradient(at), Inf)
@test norm(JuLIP.gradient(at), Inf) < 1e-4
@info "note it is correct that virial is O(1) since we applied pressure"
