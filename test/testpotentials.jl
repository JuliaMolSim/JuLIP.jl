
using JuLIP
using JuLIP.Potentials
using JuLIP.Testing

using LinearAlgebra


pairpotentials = [
   LennardJones(1.0,1.0);
   Morse(4.0,1.0,1.0);
   SWCutoff(1.0, 3.0) * LennardJones(1.0,1.0);
   SplineCutoff(2.0, 3.0) * LennardJones(1.0,1.0);
   LennardJones(1.0, 1.0) * C2Shift(2.0);
   ZBLPotential(5, 8)
]

if eam_W4 != nothing
   push!(pairpotentials, eam_W4.ϕ)
end

h2("Testing pair potential implementations")
r = range(0.8, stop=4.0, length=100) |> collect
push!(r, 2.0-1e-12)
for pp in pairpotentials
   h3(pp)
   println(@test fdtest(pp, r, verbose=verbose))
end

h2("testing shift-cutoffs: ")
V = @analytic r -> exp(r)
Vhs = V * HS(1.0)
r1 = range(0.0, stop=1.0-1e-14, length=20)
r2 = range(1.0+1e-14, stop=3.0, length=20)
h3("HS")
println(@test Vhs.(r1) == exp.(r1))
println(@test norm(Vhs.(r2)) == 0.0)
h3("V0")
V0 = V * C0Shift(1.0)
println(@test V0.(r1) ≈ exp.(r1) .- exp(1.0))
println(@test norm(V0.(r2)) == 0.0)
h3("V1")
V1 = V * C1Shift(1.0)
println(@test V1.(r1) ≈ exp.(r1) .- exp(1.0) .- exp(1.0) .* (r1.-1.0))
println(@test norm(V1.(r2)) == 0.0)
h3("V2")
V2 = V * C2Shift(1.0)
println(@test V2.(r1) ≈ exp.(r1) .- exp(1.0) .- exp(1.0) .* (r1.-1.0) .- 0.5 .* exp(1.0) .* (r1.-1.0).^2)
println(@test norm(V2.(r2)) == 0.0)


# =============================================================

calculators = Any[]

# Basic lennard-jones calculator test
push!(calculators,
      (lennardjones(r0=rnn(:Al)),
       bulk(:Al, cubic=true, pbc=(true,false,false)) * (3,3,2) ) )

# TODO PROBABLY NEED TO REVISIT THIS ONE
# # [3] JuLIP's EMT calculator
# at2 = set_pbc!( bulk(:Cu, cubic=true) * (2,2,2), (true,false,false) )
# set_positions!(at2, positions(at))
# emt2 = JuLIP.Potentials.EMTCalculator(at2)
# set_calculator!(at2, emt2)
# push!(calculators, (emt2, at2))

# ZBL Calculator
push!(calculators,
      ( ZBLPotential(4, 7) * SplineCutoff(6.0, 8.0),
        rattle!(bulk(:W, cubic=true, pbc=false) * (3,3,2), 0.1) ) )

# Stillinger-Weber model
at3 = set_pbc!( bulk(:Si, cubic=true) * 2, (false, true, false) )
sw = StillingerWeber()
set_calculator!(at3, sw)
push!(calculators, (sw, at3))

# EAM Potential
at9 = set_pbc!( bulk(:Fe, cubic = true), false ) * 2
eam = eam_Fe
push!(calculators, (eam, at9))

if eam_W4 != nothing
   # Another EAM Potential
   at10 = set_pbc!( bulk(:W, cubic = true), false ) * 2
   push!(calculators, (eam_W4, at10))
end

if eam_W != nothing   # finnis-sinclair
   at11 = set_pbc!( bulk(:W, cubic = true), false ) * 2
   push!(calculators, (eam_W, at11))
end


# ========== Run the finite-difference tests for all calculators ============

h2("Testing calculator implementations")
for (calc, at_) in calculators
   println("--------------------------------")
   h3(typeof(calc))
   @show length(at_)
   println(@test fdtest(calc, at_, verbose=true))
   println("--------------------------------")
end


# ========== Test correct implementation of site_energy ============
#            and of partial_energy

h2("Testing `site_energy` and `partial_energy` ...")
at = bulk(:Si, pbc=true, cubic=true) * 3
sw = StillingerWeber()
atsm = bulk(:Si, pbc = true)
println("checking site energy identity . . .")
println(@test abs( JuLIP.Potentials.site_energy(sw, at, 1) - energy(sw, atsm) / 2 ) < 1e-10)
rattle!(at, 0.01)
println(@test abs( energy(sw, at) - sum(site_energies(sw, at)) ) < 1e-10)

println("fd test for site_energy")
f(x) = JuLIP.Potentials.site_energy(sw, set_dofs!(at, x), 1)
df(x) = (JuLIP.Potentials.site_energy_d(sw, set_dofs!(at, x), 1) |> mat)[:]
println(@test fdtest(f, df, dofs(at); verbose=true))

println("fd test for partial energy")
Idom = [2,4,10]
f(x) = JuLIP.Potentials.energy(sw, set_dofs!(at, x); domain = Idom)
df(x) = - (JuLIP.Potentials.forces(sw, set_dofs!(at, x); domain = Idom) |> mat)[:]
println(@test fdtest(f, df, dofs(at); verbose=true))
