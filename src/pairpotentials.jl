# included from Potentials.jl
# part of the module JuLIP.Potentials

using JuLIP: zerovecs, JVecsF, JVecF, JMatF
using JuLIP.ASE.MatSciPy: NeighbourList

export ZeroPairPotential, PairSitePotential,
         LennardJones, lennardjones,
         Morse, morse

function site_energies(pp::PairPotential, at::AbstractAtoms)
   Es = zeros(length(at))
   for (i,_2,r,_3,_4) in bonds(at, cutoff(pp))
      Es[i] += pp(r)
   end
   return Es
end


# TODO: why is this not in the abstract file?
# a simplified way to calculate gradients of pair potentials
grad(pp::PairPotential, r::Float64, R::JVecF) =
            (evaluate_d(pp::PairPotential, r) / r) * R

# TODO: rewrite using generator once bug is fixed
function energy(pp::PairPotential, at::AbstractAtoms)
   E = 0.0
   for (_₁, _₂, r, _₃, _₄) in bonds(at, cutoff(pp))
      E += pp(r)
   end
   return E
end

function forces(pp::PairPotential, at::AbstractAtoms)
   dE = zerovecs(length(at))
   for (i,j,r,R,_) in bonds(at, cutoff(pp))
      # TODO: this should be equivalent, but for some reason using @GRAD is much slower!
      # dE[j] -= 2 * (@GRAD pp(r, R))    # ∇ϕ(|R|) = (ϕ'(r)/r) R
      dE[j] -= 2 * grad(pp, r, R)
   end
   return dE
end


# TODO: rewrite using generator once bug is fixed
function virial(pp::PairPotential, at::AbstractAtoms)
   S = zero(JMatF)
   for (_₁, _₂, r, R, _₃) in bonds(at, cutoff(pp))
      S -= grad(pp, r, R) * R'  # (((@D pp(r)) / r) * R) * R'
   end
   return S
end


hess(pp::PairPotential, r::Float64, R::JVecF) = (
      evaluate_dd(pp, r) * (R * R')
      + evaluate_d(pp, r)/r * ((@SMatrix eye(3)) - R * R')
  )

# hess(pp::PairPotential, r::Float64, R::JVecF) = (
#         (@DD pp(r)) * (R * R')
#         + (@D pp(r))/r * ((@SMatrix eye(3)) - R * R')
#     )

function hessian_pos(pp::PairPotential, r, R)
  # TODO: assemble the global hessian here
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


@pot  type ZeroPairPotential end
"""
`ZeroPairPotential()`: creates a potential that just returns zero
""" ZeroPairPotential
evaluate(p::ZeroPairPotential, r::Float64) = 0.0
evaluate_d(p::ZeroPairPotential, r::Float64) = 0.0
evaluate_dd(p::ZeroPairPotential, r::Float64) = 0.0
cutoff(p::ZeroPairPotential) = 0.0


# ========================================================
# wrapping a pair potential in a site potential

SitePotential(pp::PairPotential) = PairSitePotential(pp)

@pot type PairSitePotential{P} <: SitePotential
   pp::P
end

cutoff(psp::PairSitePotential) = cutoff(psp.pp)

function _sumpair_(pp, r)
   # cant use a generator here since type is not inferred!
   # Watch out for a bugfix
   s = 0.0
   for t in r
      s += pp(t)
   end
   return s
end

# special implementation of site energy and forces for a plain pair potential
evaluate(psp::PairSitePotential, r, R) = _sumpair_(psp.pp, r)

evaluate_d(psp::PairSitePotential, r, R) =
            [ grad(psp.pp, s, S) for (s, S) in zip(r, R) ]


# TODO: define a `ComposePotential?`
# and construct e.g. with  F ∘ sitepot = ComposePot(F, sitepot)



# ======================================================================
#      Special Preconditioner for Pair Potentials
# ======================================================================
