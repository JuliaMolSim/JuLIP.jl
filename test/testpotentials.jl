
using JuLIP
using JuLIP.Potentials
using JuLIP.Testing
using JuLIP.ASE


pairpotentials = [
   LennardJones(1.0,1.0);
   Morse(4.0,1.0,1.0);
   SWCutoff(1.0, 3.0) * LennardJones(1.0,1.0);
   SplineCutoff(2.0, 3.0) * LennardJones(1.0,1.0);
]

println("============================================")
println("  Testing pair potential implementations ")
println("============================================")
r = linspace(0.8, 4.0, 100)
for pp in pairpotentials
   println("--------------------------------")
   println(pp)
   println("--------------------------------")
   fdtest(pp, r, verbose=verbose)
end

# =============================================================

calculators = Any[]

# [1] basic lennard-jones calculator test
push!(calculators, (  lennardjones(r0=rnn("Al")),
         Atoms("Al", cubic=true, repeatcell=(3,3,2), pbc=(true,false,false)) ) )

# [2] ASE's EMT calculator
emt = JuLIP.ASE.EMTCalculator()
at = Atoms("Cu", cubic=true, repeatcell=(2,2,2); pbc=(true,false,false))
rattle!(at, 0.1)
set_calculator!(at, emt)
push!(calculators, (emt, at))

# [3] JuLIP's EMT calculator
at2 = Atoms("Cu", cubic=true, repeatcell=(2,2,2); pbc=(true,false,false))
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
@test abs(energy(at) - energy(at2)) < 1e-12

# [4] Stillinger-Weber model
at3 = Atoms("Si", cubic=true, pbc=(false, true, false)) * 2
sw = StillingerWeber()
set_calculator!(at3, sw)
push!(calculators, (sw, at3))

# [5] a simple FDPotential
@pot type FDPot <: FDPotential end
fdpot(r) = exp(-0.3*r) * JuLIP.Potentials.cutsw(r, 4.0, 1.0)
JuLIP.Potentials.ad_evaluate{T<:Real}(pot::FDPot, R::Matrix{T}) =
               sum( fdpot(Base.LinAlg.vecnorm2(R[:,i])) for i = 1:size(R,2) )
JuLIP.cutoff(::FDPot) = 4.0
at5 = Atoms("Si", pbc=false) * (3,3,1)
push!(calculators, (FDPot(), at5))

# [6] a simple FDPotential
@pot type FDPot_r <: FDPotential_r end
JuLIP.Potentials.ad_evaluate{T<:Real}(pot::FDPot_r, r::Vector{T}) = sum( fdpot.(r) )
JuLIP.cutoff(::FDPot_r) = 4.0
at6 = Atoms("Si", pbc=false) * (3,3,1)
push!(calculators, (FDPot_r(), at6))

# [7] a simple FDPotential
@pot type RDPot_r <: RDPotential_r end
JuLIP.Potentials.ad_evaluate{T<:Real}(pot::RDPot_r, r::Vector{T}) = sum( fdpot.(r) )
JuLIP.cutoff(::RDPot_r) = 4.0
at7 = Atoms("Si", pbc=false) * (3,3,1)
push!(calculators, (RDPot_r(), at7))

# [8] PairSitePotential
at8 = Atoms("Al", cubic=true, pbc=false) * 2
pp = lennardjones(r0=rnn("Al"))
psp = SitePotential(pp)
push!(calculators, (psp, at8))

println("--------------------------------------------------")
println(" PairSitePotential Consistency test: ")
println("--------------------------------------------------")
println(" E_pp - E_psp = ", energy(pp, at8) - energy(psp, at8))
println(" |Frc_pp - Frc_psp = ", maxnorm(forces(pp, at8) - forces(psp, at8)))
println("--------------------------------------------------")
@test abs(energy(pp, at8) - energy(psp, at8)) < 1e-12



# ========== Run the finite-difference tests for all calculators ============

println("============================================")
println("  Testing calculator implementations ")
println("============================================")
for (calc, at) in calculators
   println("--------------------------------")
   println(typeof(calc))
   @show length(at)
   println("--------------------------------")
   fdtest(calc, at, verbose=true)
end
