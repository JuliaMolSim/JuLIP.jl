# included from Potentials.jl
# part of the module JuLIP.Potentials

import JuLIP: zerovecs, energy, forces
import JuLIP.ASE.MatSciPy: NeighbourList

export ZeroPairPotential, PairCalculator


# a simplified way to calculate gradients of pair potentials
grad(p::PairPotential, r::Float64, R::JVec) =
            (evaluate_d(p::PairPotential, r) / r) * R


"`PairCalculator` : basic calculator for pair potentials."
type PairCalculator{T <: PairPotential} <: AbstractCalculator
    pp::T
end

cutoff(calc::PairCalculator) = cutoff(calc.pp)

function energy(calc::PairCalculator, at::AbstractAtoms)
   E = 0.0
   for (_,_,r,_,_) in bonds(at, cutoff(calc))
      E += calc.pp(r)
   end
   return E
end

function forces(calc::PairCalculator, at::AbstractAtoms)
   dE = zerovecs(length(at))
   for (i,j,r,R,_) in bonds(at, cutoff(calc))
      # TODO: this should be equivalent, but for some reason using @GRAD is much slower!
      # dE[j] -= 2 * @GRAD calc.pp(r, R)   # ∇ϕ(|R|) = (ϕ'(r)/r) R
      dE[j] -= 2 * ((@D calc.pp(r))/r) * R
   end
   return dE
end



"""
`LennardJonesPotential:` e0 * ( (r0/r)¹² - 2 (r0/r)⁶ )

Constructor: `LennardJonesPotential(;r0=1.0, e0=1.0)`
"""
LennardJonesPotential(; r0=1.0, e0=1.0) =
   AnalyticPotential(:($e0 * ((r/$r0)^(-12) - 2.0 * (r/$r0)^(-6))),
                     id = "LennardJones(r0=$r0, e0=$e0)")

LennardJonesCalculator(;r0=1.0, e0=1.0, rcut= (1.9*r0, 2.7*r0)) =
   PairCalculator( SplineCutoff(rcut[1], rcut[2]) *
                           LennardJonesPotential(r0=r0, e0=e0) )



"""
`MorsePotential(;A=4.0, e0=1.0, r0=1.0)`: constructs an
`AnalyticPotential` for
   e0 ( exp( -2 A (r/r0 - 1) ) - 2 exp( - A (r/r0 - 1) ) )
"""
MorsePotential(;A=4.0, e0=1.0, r0=1.0) =
   AnalyticPotential(:( $e0 * ( exp(-$(2.0*A) * (r/$r0 - 1.0))
                               - 2.0 * exp(-$A * (r/$r0 - 1.0)) ) ),
                     id="MorsePotential(A=$A, e0=$e0, r0=$r0)")

MorseCalculator(;A=4.0, e0=1.0, r0=1.0, rcut= (1.9*r0, 2.7*r0)) =
   PairCalculator( SplineCutoff(rcut[1], rcut[2]) *
                           MorsePotential(A=A, r0=r0, e0=e0) )



# """
# `ZeroPairPotential()`: creates a potential that just returns zero
# """
@pot  type ZeroPairPotential end
evaluate(p::ZeroPairPotential, r::Float64) = 0.0
evaluate_d(p::ZeroPairPotential, r::Float64) = 0.0
evaluate_dd(p::ZeroPairPotential, r::Float64) = 0.0
cutoff(p::ZeroPairPotential) = 0.0


# ======================================================================
#      Preconditioner for Pair Potentials
# ======================================================================
