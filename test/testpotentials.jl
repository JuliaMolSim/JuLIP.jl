
using JuLIP
using JuLIP.Potentials
using JuLIP.Testing
using JuLIP.ASE


pairpotentials = [
   LennardJones(1.0,1.0);
   Morse(4.0,1.0,1.0);
   SWCutoff(1.0, 3.0) * LennardJones(1.0,1.0);
   SplineCutoff(2.0, 3.0) * LennardJones(1.0,1.0);
   LennardJones(1.0, 1.0) * C2Shift(2.0);
]

println("============================================")
println("  Testing pair potential implementations ")
println("============================================")
r = linspace(0.8, 4.0, 100) |> collect
push!(r, 2.0-1e-12)
for pp in pairpotentials
   println("--------------------------------")
   println(pp)
   println("--------------------------------")
   @test fdtest(pp, r, verbose=verbose)
end

print("testing shift-cutoffs: ")
V = @analytic r -> exp(r)
Vhs = V * HS(1.0)
r1 = linspace(0.0, 1.0-1e-14, 20)
r2 = linspace(1.0+1e-14, 3.0, 20)
print("HS")
@test Vhs.(r1) == exp.(r1)
@test norm(Vhs.(r2)) == 0.0
print(", V0")
V0 = V * C0Shift(1.0)
@test V0.(r1) ≈ exp.(r1) - exp(1.0)
@test norm(V0.(r2)) == 0.0
print(", V1")
V1 = V * C1Shift(1.0)
@test V1.(r1) ≈ exp.(r1) - exp(1.0) - exp(1.0) * (r1-1.0)
@test norm(V1.(r2)) == 0.0
println(", V2")
V2 = V * C2Shift(1.0)
@test V2.(r1) ≈ exp.(r1) - exp(1.0) - exp(1.0) * (r1-1.0) - 0.5 * exp(1.0) * (r1-1.0).^2
@test norm(V2.(r2)) == 0.0


# =============================================================

calculators = Any[]

# [1] basic lennard-jones calculator test
push!(calculators, (  lennardjones(r0=rnn("Al")),
         set_pbc!(bulk("Al", cubic=true) * (3,3,2), (true,false,false)) ) )

# [2] ASE's EMT calculator
emt = JuLIP.ASE.EMTCalculator()
at = rattle!( set_pbc!( bulk("Cu", cubic=true) * 2, (true,false,false) ), 0.1 )
set_calculator!(at, emt)
if notCI
   push!(calculators, (emt, at))
end

# [3] JuLIP's EMT calculator
at2 = set_pbc!( bulk("Cu", cubic=true) * (2,2,2), (true,false,false) )
set_positions!(at2, positions(at))
emt2 = JuLIP.Potentials.EMTCalculator(at2)
set_calculator!(at2, emt2)
push!(calculators, (emt2, at2))

println("--------------------------------------------------")
println(" EMT Consistency test: ")
println("--------------------------------------------------")
println(" E_ase - E_jl = ", energy(at) - energy(at2))
println(" |Frc_ase - Frc_jl = ", maxnorm(forces(at) - forces(at2)))
println("--------------------------------------------------")
@test abs(energy(at) - energy(at2)) < 1e-10

# [4] Stillinger-Weber model
at3 = set_pbc!( bulk("Si", cubic=true) * 2, (false, true, false) )
sw = StillingerWeber()
set_calculator!(at3, sw)
push!(calculators, (sw, at3))

# ================================================================
# # [5] a simple FDPotential
# @pot type FDPot <: FDPotential end
# fdpot(r) = exp(-0.3*r) * JuLIP.Potentials.cutsw(r, 4.0, 1.0)
# JuLIP.Potentials.ad_evaluate{T<:Real}(pot::FDPot, R::Matrix{T}) =
#                sum( fdpot(Base.LinAlg.vecnorm2(R[:,i])) for i = 1:size(R,2) )
# JuLIP.cutoff(::FDPot) = 4.0
# at5 = set_pbc!(bulk("Si") * (3,3,1), false)
# push!(calculators, (FDPot(), at5))
#
# # [6] a simple FDPotential
# @pot type FDPot_r <: FDPotential_r end
# JuLIP.Potentials.ad_evaluate{T<:Real}(pot::FDPot_r, r::Vector{T}) = sum( fdpot.(r) )
# JuLIP.cutoff(::FDPot_r) = 4.0
# at6 = set_pbc!(bulk("Si") * (3,3,1), false)
# push!(calculators, (FDPot_r(), at6))
#
# # [7] a simple RDPotential
# @pot type RDPot_r <: RDPotential_r end
# JuLIP.Potentials.ad_evaluate{T<:Real}(pot::RDPot_r, r::Vector{T}) = sum( fdpot.(r) )
# JuLIP.cutoff(::RDPot_r) = 4.0
# at7 = set_pbc!(bulk("Si") * (3,3,1), false)
# push!(calculators, (RDPot_r(), at7))
# ================================================================


# [8] PairSitePotential
at8 = set_pbc!( bulk("Al", cubic=true), false ) * 2
pp = lennardjones(r0=rnn("Al"))
psp = SitePotential(pp)
if notCI
   push!(calculators, (psp, at8))
end

println("--------------------------------------------------")
println(" PairSitePotential Consistency test: ")
println("--------------------------------------------------")
println(" E_pp - E_psp = ", energy(pp, at8) - energy(psp, at8))
println(" |Frc_pp - Frc_psp = ", maxnorm(forces(pp, at8) - forces(psp, at8)))
println("--------------------------------------------------")
@test abs(energy(pp, at8) - energy(psp, at8)) < 1e-11


# [9] EAM Potential
at9 = set_pbc!( bulk("Fe", cubic = true), false ) * (2,1,1)
eam = eam_Fe
push!(calculators, (eam, at9))

if eam_W4 != nothing
   # [10] Another EAM Potential
   at10 = set_pbc!( bulk("W", cubic = true), false ) * (2,1,2)
   eam4 = eam_W4
   push!(calculators, (eam4, at10))
end

# ========== Run the finite-difference tests for all calculators ============

println("============================================")
println("  Testing calculator implementations ")
println("============================================")
for (calc, at) in calculators
   println("--------------------------------")
   println(typeof(calc))
   @show length(at)
   println("--------------------------------")
   @test fdtest(calc, at, verbose=true)
end

quit()

# ========== Test correct implementation of site_energy ============

println("Testing `site_energy` ...")
at = bulk("Si", pbc=true, cubic=true) * 3
sw = StillingerWeber()
atsm = bulk("Si", pbc = true)
println(" ... passed site_energy identity, now testing derivative ...")
@test abs( JuLIP.Potentials.site_energy(sw, at, 1) - energy(sw, atsm) / 2 ) < 1e-10

# finite-difference test
set_constraint!(at, FixedCell(at))
f(x) = JuLIP.Potentials.site_energy(sw, set_dofs!(at, x), 1)
df(x) = (JuLIP.Potentials.site_energy_d(sw, set_dofs!(at, x), 1) |> mat)[:]
@test fdtest(f, df, dofs(at); verbose=true)
