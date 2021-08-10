
using JuLIP
using JuLIP.Potentials
using JuLIP.Testing
using LinearAlgebra

##
pairpotentials = [
   ("LennardJones", LennardJones(1.0,1.0)),
   ("Morse", Morse(4.0,1.0,1.0)),
   ("SWCutoff * LennardJones", SWCutoff(1.0, 3.0) * LennardJones(1.0,1.0)),
   ("SplineCutoff * LennardJones", SplineCutoff(2.0, 3.0) * LennardJones(1.0,1.0)),
   ("LennardJones * C2Shift", LennardJones(1.0, 1.0) * C2Shift(2.0)),
   ("EAM.rho", eam_W4.ρ[1])
]

h2("Testing pair potential implementations")
r = range(0.8, stop=4.0, length=100) |> collect
push!(r, 2.0-1e-12)
for (name, pp) in pairpotentials
   h3(name)
   println(@test fdtest(pp, r, verbose=verbose))
end

h2("testing shift-cutoffs: ")
V = @analytic r -> exp(r)
Vhs = V * HS(1.0)
r1 = range(0.0, stop=1.0-1e-14, length=20)
r2 = range(1.0+1e-14, stop=3.0, length=20)
h3("HS")
println(@test Vhs.(r1) ≈ exp.(r1))
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


## =============================================================

calculators = Any[]

# Basic lennard-jones calculator test
push!(calculators,
      (lennardjones(r0=rnn(:Al)),
       bulk(:Al, cubic=true, pbc=(true,false,false)) * (3,3,2) ) )

# ZBL Calculator
push!(calculators,
      ( ZBLPotential() * SplineCutoff(6.0, 8.0),
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

# Another EAM Potential
at10 = set_pbc!( bulk(:W, cubic = true), false ) * 2
push!(calculators, (eam_W4, at10))

at = bulk(:Pd) * 3
at.Z[1:3:end] .= AtomicNumber(:Ag)
at.Z[1:5:end] .= AtomicNumber(:H)
rattle!(at, 0.1)
push!(calculators, (eam_PdAgH, at))

# JuLIP's EMT implementation
at = set_pbc!( bulk(:Cu, cubic=true) * (2,2,2), (true,false,false) )
rattle!(at, 0.02)
emt = EMT(:Cu)
push!(calculators, (emt, at))

# and a multi-species EMT
at1 = bulk(:Cu, cubic=true) * 2
at1.Z[[2,4]] .= atomic_number(:Al)
at1.Z[[8, 9, 10]] .= atomic_number(:Ni)
at1 = set_pbc!(at1, false)
rattle!(at1, 0.02)
emt = EMT()
push!(calculators, (emt, at1))


## ========== Run the finite-difference tests for all calculators ============

h2("Testing calculator implementations")
for (calc, at_) in calculators
   println("--------------------------------")
   h3(typeof(calc))
   @show length(at_)
   println(@test fdtest(calc, at_, verbose=true))
   println("--------------------------------")
end

##

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
