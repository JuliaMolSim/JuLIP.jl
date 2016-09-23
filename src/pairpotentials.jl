# included from Potentials.jl
# part of the module JuLIP.Potentials

using JuLIP: zerovecs, JVecsF, JVecF
using JuLIP.ASE.MatSciPy: NeighbourList

export ZeroPairPotential, PairSitePotential,
         LennardJones, lennardjones,
         Morse, morse


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

Constructor: `LennardJonesPotential(r0, e0)`
"""
LennardJones(r0, e0) =
   AnalyticPotential(:($e0 * (($r0/r)^(12) - 2.0 * ($r0/r)^(6))),
                     id = "LennardJones(r0=$r0, e0=$e0)")

LennardJones() = LennardJones(1.0, 1.0)

"""
`lennardjones(; r0=1.0, e0=1.0, rcut = (1.9*r0, 2.7*r0))`

simplified constructor for `LennardJones` (type unstable)
"""
lennardjones(; r0=1.0, e0=1.0, rcut = (1.9*r0, 2.7*r0)) = (
   (rcut == nothing || rcut == Inf)
         ?  LennardJones(r0, e0)
         :  SplineCutoff(rcut[1], rcut[2]) * LennardJones(r0, e0) )


"""
`Morse(A, e0, r0)`: constructs an
`AnalyticPotential` for
   e0 ( exp( -2 A (r/r0 - 1) ) - 2 exp( - A (r/r0 - 1) ) )
"""
Morse(A, e0, r0) =
   AnalyticPotential(:( $e0 * ( exp(-$(2.0*A) * (r/$r0 - 1.0))
                                - 2.0 * exp(-$A * (r/$r0 - 1.0)) ) ),
                     id="MorsePotential(A=$A, e0=$e0, r0=$r0)")
Morse(;A=4.0, e0=1.0, r0=1.0) = Morse(A, e0, r0)

"""
`morse(A=4.0, e0=1.0, r0=1.0, rcut=(1.9*r0, 2.7*r0))`

simplified constructor for `Morse` (type unstable)
"""
morse(A=4.0, e0=1.0, r0=1.0, rcut=(1.9*r0, 2.7*r0)) = (
   (rcut == nothing || rcut == Inf)
         ?  Morse(A, e0, r0)
         :  SplineCutoff(rcut[1], rcut[2]) * Morse(A, e0, r0) )


@pot type ZeroPairPotential end
"""
`ZeroPairPotential()`: creates a potential that just returns zero
"""
ZeroPairPotential
evaluate(p::ZeroPairPotential, r::Float64) = 0.0
evaluate_d(p::ZeroPairPotential, r::Float64) = 0.0
evaluate_dd(p::ZeroPairPotential, r::Float64) = 0.0
cutoff(p::ZeroPairPotential) = 0.0


# ========================================================
# wrapping a pair potential in a site potential

SitePotential(pp::PairPotential) = PairSitePotential(pp, Val{:one}())

Base.zero(::Type{JVecF}) = JVecF(0.0,0.0,0.0)

@pot type PairSitePotential{PT, FT} <: SitePotential
   pp::PT
   F::FT
end

cutoff(psp::PairSitePotential) = cutoff(psp.pp)

function _sumpair_(pp, r)
   # cant use a generator here since type is not inferred!
   # Watch out for a bugfix
   s = 0.0
   for s in r
      s += psp.pp(s)
   end
   return s
end

# special implementation of site energy and forces for a plain pair potential
evaluate{PT}(psp::PairSitePotential{PT,Val{:one}}, r, R) = _sumpair_(psp.pp, r)

evaluate_d{PT}(psp::PairSitePotential{PT,Val{:one}}, r, R) =
            [ ((@D psp.pp(s))/s) * S for (s, S) in zip(r, R) ]

# general implementation with a nonlinear wrapper
evaluate(psp::PairSitePotential, r, R) = psp.F(_sumpair_(psp.pp, r))

function evaluate_d{PT}(psp::PairSitePotential{PT,Val{:one}}, r, R)
   dF = @D psp.F(_sumpair_(psp.pp, r))
   return [ dF_ * ((@D psp.pp(s))/s) * S for (s, S, dF_) in zip(r, R, dF) ]
end

# instead of this, use a `ComposePotential?`
# and construct e.g. with  F ∘ sitepot = ComposePot(F, sitepot)




# ======================================================================
#      Special Preconditioner for Pair Potentials
# ======================================================================
