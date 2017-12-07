# included from Potentials.jl
# part of the module JuLIP.Potentials

using JuLIP: zerovecs, JVecsF, JVecF, JMatF
using JuLIP.ASE.MatSciPy: NeighbourList

export ZeroPairPotential, PairSitePotential,
         LennardJones, lennardjones,
         Morse, morse, grad, hess

function site_energies(pp::PairPotential, at::AbstractAtoms)
   Es = zeros(length(at))
   for (i,_2,r,_3,_4) in bonds(at, cutoff(pp))
      Es[i] += 0.5 * pp(r)
   end
   return Es
end

import Base.sum
sum(V::PairPotential, r) = sum( V(s) for s in r )
function sum(V::PairPotential, r::Vector{T}) where T <: Real
   E = 0.0
   @simd for n = 1:length(r)
      @inbounds E += V(r[n])
   end
   return 0.5 * E
end

# a simplified way to calculate gradients of pair potentials
@inline grad(V::PairPotential, r::Real, R::JVec) = (evaluate_d(V, r) / r) * R

function energy(V::PairPotential, at::AbstractAtoms)
   E = 0.0
   for (_₁, _₂, r, _₃, _₄) in bonds(at, cutoff(V))
      E += V(r)
   end
   return 0.5 * E
end

using JuLIP.ASE: ASEAtoms

function energy(V::PairPotential, at::ASEAtoms)
   nlist = neighbourlist(at, cutoff(V))
   return 0.5 * sum(V.(nlist.r))
end


function forces(V::PairPotential, at::AbstractAtoms)
   dE = zerovecs(length(at))
   for (i,j,r,R,_) in bonds(at, cutoff(V))
      dE[i] += @GRAD V(r, R)
   end
   return dE
end

function forces(V::PairPotential, at::ASEAtoms)
   nlist = neighbourlist(at, cutoff(V))
   dE = zerovecs(length(at))
   @simd for n = 1:length(nlist)
      @inbounds dE[nlist.i[n]] += grad(V, nlist.r[n], nlist.R[n])
   end
   return dE
end


# TODO: rewrite using generator once bug is fixed (???or maybe decide not to bother???)
function virial(pp::PairPotential, at::AbstractAtoms)
   S = zero(JMatF)
   for (_₁, _₂, r, R, _₃) in bonds(at, cutoff(pp))
      S -= 0.5 * grad(pp, r, R) * R'  # (((@D pp(r)) / r) * R) * R'
   end
   return S
end


function hess(V::PairPotential, r::Float64, R::JVecF)
   R̂ = R/r
   P = R̂ * R̂'
   dV = (@D V(r))/r
   return ((@DD V(r)) - dV) * P + dV * eye(JMatF)
end


function hessian_pos(pp::PairPotential, at::AbstractAtoms)
   nlist = neighbourlist(at, cutoff(pp))
   I, J, Z = Int[], Int[], JMatF[]
   for C in (I, J, Z); sizehint!(C, 2*length(nlist)); end
   for (i, j, r, R, _) in bonds(nlist)
      h = 0.5 * hess(pp, r, R)
      append!(I, (i,  i,  j, j))
      append!(J, (i,  j,  i, j))
      append!(Z, (h, -h, -h, h))
   end
   hE = sparse(I, J, Z, length(at), length(at))
   return hE
end



"""
`LennardJones(r0, a0)` or `LennardJones(;r0=1.0, e0=1.0)` :

construct the Lennard-Jones potential e0 * ( (r0/r)¹² - 2 (r0/r)⁶ )
"""
LennardJones(r0, e0) = @analytic r -> e0 * 4.0 * ((r0/r)^(12) - (r0/r)^(6))
LennardJones(;r0=1.0, e0=1.0) = LennardJones(r0, e0)

"""
`lennardjones(; r0=1.0, e0=1.0, rcut = (1.9*r0, 2.7*r0))`

simplified constructor for `LennardJones` (type unstable)
"""
lennardjones(; r0=1.0, e0=1.0, rcut = (1.9*r0, 2.7*r0)) = (
   (rcut == nothing || rcut == Inf)
         ?  LennardJones(r0, e0)
         :  SplineCutoff(rcut[1], rcut[2]) * LennardJones(r0, e0) )


"""
`Morse(A, e0, r0)` or `Morse(;A=4.0, e0=1.0, r0=1.0)`: constructs a
`PairPotential` for
   e0 ( exp( -2 A (r/r0 - 1) ) - 2 exp( - A (r/r0 - 1) ) )
"""
Morse(A, e0, r0) = @analytic(
   r -> e0 * ( exp(-(2.0*A) * (r/r0 - 1.0)) - 2.0 * exp(-A * (r/r0 - 1.0)) ) )
Morse(;A=4.0, e0=1.0, r0=1.0) = Morse(A, e0, r0)

"""
`morse(A=4.0, e0=1.0, r0=1.0, rcut=(1.9*r0, 2.7*r0))`

simplified constructor for `Morse` (type unstable)
"""
morse(;A=4.0, e0=1.0, r0=1.0, rcut=(1.9*r0, 2.7*r0)) = (
   (rcut == nothing || rcut == Inf)
         ?  Morse(A, e0, r0)
         :  SplineCutoff(rcut[1], rcut[2]) * Morse(A, e0, r0) )


@pot struct ZeroPairPotential end
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

@pot struct PairSitePotential{P} <: SitePotential
   pp::P
end

cutoff(psp::PairSitePotential) = cutoff(psp.pp)

# TODO: get rid of this!
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
evaluate(psp::PairSitePotential, r, R) = 0.5 * _sumpair_(psp.pp, r)

evaluate_d(psp::PairSitePotential, r, R) =
            [ 0.5 * grad(psp.pp, s, S) for (s, S) in zip(r, R) ]


# an FF preconditioner for pair potentials
function precon(V::PairPotential, r, R)
   dV = 0.5 * (@D V(r))
   hV = 0.5 * (@DD V(r))
   S = R/r
   return 0.9 * (abs(hV) * S * S' + abs(dV / r) * (eye(JMatF) - S * S')) +
          0.1 * (abs(hV) + abs(dV / r)) * eye(JMatF)
end


# TODO: define a `ComposePotential?`
# and construct e.g. with  F ∘ sitepot = ComposePot(F, sitepot)
