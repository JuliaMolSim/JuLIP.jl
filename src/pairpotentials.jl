# included from Potentials.jl
# part of the module JuLIP.Potentials

using JuLIP: zerovecs
using JuLIP.ASE.MatSciPy: NeighbourList

import JuLIP: energy, forces

export ZeroPairPotential, LennardJones, Morse


# TODO: why is this not in the abstract file?
# a simplified way to calculate gradients of pair potentials
grad(pp::PairPotential, r::Float64, R::JVec) =
            (evaluate_d(p::PairPotential, r) / r) * R

energy(pp::PairPotential, at::AbstractAtoms) =
         sum( pp(r) for (_1,_2,r,_3,_4) in bonds(at, cutoff(pp)) )

function forces(pp::PairPotential, at::AbstractAtoms)
   dE = zerovecs(length(at))
   for (i,j,r,R,_) in bonds(at, cutoff(pp))
      # TODO: this should be equivalent, but for some reason using @GRAD is much slower!
      # dE[j] -= 2 * @GRAD calc.pp(r, R)   # ∇ϕ(|R|) = (ϕ'(r)/r) R
      dE[j] -= 2 * ((@D pp(r))/r) * R
   end
   return dE
end



"""
`LennardJones:` e0 * ( (r0/r)¹² - 2 (r0/r)⁶ )

Constructor: `LennardJonesPotential(;r0=1.0, e0=1.0)`
"""
LennardJones(; r0=1.0, e0=1.0) =
   AnalyticPotential(:($e0 * (($r0/r)^(12) - 2.0 * ($r0/r)^(6))),
                     id = "LennardJones(r0=$r0, e0=$e0)")

# LennardJones(; r0=1.0, e0=1.0, rcut = (1.9*r0, 2.7*r0)) = (
#    rcut == nothing
#          ?  AnalyticPotential(:($e0 * (($r0/r)^(12) - 2.0 * ($r0/r)^(6))),
#                                        id = "LennardJones(r0=$r0, e0=$e0)")
#          :  SplineCutoff(rcut[1], rcut[2]) *
#                LennardJonesPotential(r0=r0, e0=e0, rcut=nothing)   )


"""
`Morse(;A=4.0, e0=1.0, r0=1.0)`: constructs an
`AnalyticPotential` for
   e0 ( exp( -2 A (r/r0 - 1) ) - 2 exp( - A (r/r0 - 1) ) )
"""
Morse(;A=4.0, e0=1.0, r0=1.0) =
   AnalyticPotential(:( $e0 * ( exp(-$(2.0*A) * (r/$r0 - 1.0))
                               - 2.0 * exp(-$A * (r/$r0 - 1.0)) ) ),
                     id="MorsePotential(A=$A, e0=$e0, r0=$r0)")


               # Morse(;A=4.0, e0=1.0, r0=1.0, rcut= (1.9*r0, 2.7*r0)) = (
               #    rcut == nothing
               #          ?  AnalyticPotential(:( $e0 * ( exp(-$(2.0*A) * (r/$r0 - 1.0))
               #                                      - 2.0 * exp(-$A * (r/$r0 - 1.0)) ) ),
               #                                  id="MorsePotential(A=$A, e0=$e0, r0=$r0)")
               #          :  SplineCutoff(rcut[1], rcut[2]) *
               #                Morse(A=A, r0=r0, e0=e0, rcut=nothing)    )


# """
# `ZeroPairPotential()`: creates a potential that just returns zero
# """
@pot type ZeroPairPotential end
evaluate(p::ZeroPairPotential, r::Float64) = 0.0
evaluate_d(p::ZeroPairPotential, r::Float64) = 0.0
evaluate_dd(p::ZeroPairPotential, r::Float64) = 0.0
cutoff(p::ZeroPairPotential) = 0.0



# ======================================================================
#      Special Preconditioner for Pair Potentials
# ======================================================================
