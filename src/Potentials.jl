"""
## module Potentials

### Summary

This module implements some basic interatomic potentials in pure Julia, as well
as provides building blocks and prototypes for further implementations
The implementation is done in such a way that they can be used either in "raw"
form or withinabstractframeworks.

### Types

### `evaluate`, `evaluate_d`, `evaluate_dd`, `grad`

### The `@D`, `@DD`, `@GRAD` macros

TODO: write documentation

"""
module Potentials

using JuLIP: AbstractAtoms, AbstractNeighbourList, AbstractCalculator,
      JVec, JVecs, mat, vec, JMat, JVecF, SVec, vecs, SMat,
      positions, set_positions!
using StaticArrays: @SMatrix

using NeighbourLists

using LinearAlgebra: norm

using SparseArrays: sparse

import JuLIP: energy, forces, cutoff, virial, hessian_pos, hessian,
              site_energies, r_sum

export Potential, PairPotential, SitePotential,
     site_energy, site_energy_d, partial_energy, partial_energy_d

"""
`Potential`: genericabstractsupertype for all potential-like things
"""
abstract type Potential <: AbstractCalculator end

"""
`PairPotential`:abstractsupertype for pair potentials

Can also be used as a constructor for analytic pair potentials, e.g.,
```julia
lj = @analytic r -> r^(-12) - 2 * r^(-6)
```
"""
abstract type PairPotential <: Potential end

"""
`SitePotential`:abstractsupertype for generic site potentials
"""
abstract type SitePotential <: Potential end


include("potentials_base.jl")
# * @pot, @D, @DD, @GRAD and related things
# TODO: move this to a separate package???

NeighbourLists.sites(at::AbstractAtoms, rcut::AbstractFloat) =
      sites(neighbourlist(at, rcut))

NeighbourLists.pairs(at::AbstractAtoms, rcut::AbstractFloat) =
      pairs(neighbourlist(at, rcut))


# Implementation of a generic site potential
# ================================================

site_energies(V::SitePotential, at::AbstractAtoms) =
   Float64[ V(r, R) for (_₁, _₂, r, R) in sites(at, cutoff(V)) ]

energy(V::SitePotential, at::AbstractAtoms) = r_sum(site_energies(V, at))

evaluate(V::SitePotential, R::AbstractVector{JVecF}) = evaluate(V, norm.(R), R)

evaluate_d(V::SitePotential, R::AbstractVector{JVecF}) = evaluate_d(V, norm.(R), R)

function forces(V::SitePotential, at::AbstractAtoms)
   frc = zerovecs(length(at))
   for (i, j, r, R) in sites(at, cutoff(V))
      dV = @D V(r, R)
      for a = 1:length(j)
         frc[j[a]] -= dV[a]
      end
      frc[i] += sum(dV)
   end
   return frc
end

site_virial(dV, R) =  (
      length(R) > 0 ? (- sum( dVi * Ri' for (dVi, Ri) in zip(dV, R) ))
                    : zero(JMatF) )

virial(V::SitePotential, at::AbstractAtoms) =
      sum(  site_virial((@D V(r, R)), R)
            for (_₁, _₂, r, R) in sites(at, cutoff(V))  )


# TODO: partial_energy and partial_energy_d are not tested properly

function partial_energy(V::SitePotential, at::AbstractAtoms, Idom)
   E = 0.0
   nlist = neighbourlist(at, cutoff(V))
   for i in Idom
      j, r, R = site(nlist, i)
      E += V(r, R)
   end
   return E
end

function partial_energy_d(V::SitePotential, at::AbstractAtoms, Idom)
   F = zeros(JVecF, length(at))
   nlist = neighbourlist(at, cutoff(V))
   for i in Idom
      j, r, R = site(nlist, i)
      dV = @D V(R)
      F[j] += dV
      F[i] -= sum(dV)
   end
   return F
end

partial_energy(V::PairPotential, at::AbstractAtoms, Idom) =
      partial_energy(PairSitePotential(V), at, Idom)
partial_energy_d(V::PairPotential, at::AbstractAtoms, Idom) =
      partial_energy_d(PairSitePotential(V), at, Idom)

site_energy(V::Union{SitePotential, PairPotential}, at::AbstractAtoms, i0::Int) =
      partial_energy(V, at, [i0])
site_energy_d(V::Union{SitePotential, PairPotential}, at::AbstractAtoms, i0::Int) =
      partial_energy_d(V, at, [i0])





include("analyticpotential.jl")
# * AnalyticFunction
# * WrappedPPotential

include("cutoffs.jl")
#   * SWCutoff
#   * ShiftCutoff
#   * SplineCutoff

include("pairpotentials.jl")
# * PairCalculator
# * LennardJonesPotential
# * MorsePotential
# * SimpleExponential

include("adsite.jl")
# * ADPotential : Site potential using ForwardDiff

include("stillingerweber.jl")
# * type StillingerWeber

include("splines.jl")
include("eam.jl")
# EAM, FinnisSinclair

include("onebody.jl") 


export ZeroSitePotential

"a site potential that just returns zero"
mutable struct ZeroSitePotential <: SitePotential
end

@pot ZeroSitePotential


evaluate(p::ZeroSitePotential, r, R) = 0.0
evaluate_d(p::ZeroSitePotential, r, R) = zeros(r)   # TODO: is this a bug?
cutoff(::ZeroSitePotential) = 3.0



"""
`fd_hessian(V, R, h) -> H`

If `length(R) = N` and `length(R[i]) = d` then `H` is an N × N `Matrix{SMatrix}` with
each element a d × d static array.
"""
function fd_hessian(V::SitePotential, R::Vector{SVec{D,T}}, h) where {D,T}
   d = length(R[1])
   N = length(R)
   H = zeros( typeof(@SMatrix zeros(d, d)), N, N )
   return fd_hessian!(H, V, R, h)
end

"""
`fd_hessian!(H, V, R, h) -> H`

Fill `H` with the hessian entries; cf `fd_hessian`.
"""
function fd_hessian!(H, V::SitePotential, R::Vector{SVec{D,T}}, h) where {D,T}
   N = length(R)
   # convert R into a long vector and H into a big matrix (same part of memory!)
   Rvec = mat(R)[:]
   Hmat = zeros(N*D, N*D)   # reinterpret(T, H, (N*D, N*D))
   # now re-define ∇V as a function of a long vector (rather than a vector of SVecs)
   dV(r) = (evaluate_d(V, r |> vecs) |> mat)[:]
   # compute the hessian as a big matrix
   for i = 1:N*D
      Rvec[i] +=h
      dVp = dV(Rvec)
      Rvec[i] -= 2*h
      dVm = dV(Rvec)
      Hmat[:, i] = (dVp - dVm) / (2 * h)
      Rvec[i] += h
   end
   Hmat = 0.5 * (Hmat + Hmat')
   # convert to a block-matrix
   for i = 1:N, j = 1:N
      Ii = (i-1) * D + (1:D)
      Ij = (j-1) * D + (1:D)
      H[i, j] = SMat{D,D}(Hmat[Ii, Ij])
   end
   return H
end

function fd_hessian(calc::AbstractCalculator, at::AbstractAtoms, h)
   d = 3
   N = length(at)
   H = zeros( typeof(@SMatrix zeros(d, d)), N, N )
   return fd_hessian!(H, calc, at, h)
end


"""
`fd_hessian!{D,T}(H, calc, at, h) -> H`

Fill `H` with the hessian entries; cf `fd_hessian`.
"""
function fd_hessian!(H, calc::AbstractCalculator, at::AbstractAtoms, h)
   D = 3
   N = length(at)
   X = positions(at) |> mat
   x = X[:]
   # convert R into a long vector and H into a big matrix (same part of memory!)
   Hmat = zeros(N*D, N*D)
   # now re-define ∇V as a function of a long vector (rather than a vector of SVecs)
   dE(x_) = (site_energy_d(calc, set_positions!(at, reshape(x_, D, N)), 1) |> mat)[:]
   # compute the hessian as a big matrix
   for i = 1:N*D
      x[i] += h
      dEp = dE(x)
      x[i] -= 2*h
      dEm = dE(x)
      Hmat[:, i] = (dEp - dEm) / (2 * h)
      x[i] += h
   end
   Hmat = 0.5 * (Hmat + Hmat')
   # convert to a block-matrix
   for i = 1:N, j = 1:N
      Ii = (i-1) * D + (1:D)
      Ij = (j-1) * D + (1:D)
      H[i, j] = SMat{D,D}(Hmat[Ii, Ij])
   end
   return H
end


hessian_pos(V::SitePotential, at::AbstractAtoms) =
      _precon_or_hessian_pos(V, at, hess)

# implementation of a generic assembly of a global block-hessian from
# local site-hessians
function _precon_or_hessian_pos(V::SitePotential, at::AbstractAtoms, hfun)
   nlist = neighbourlist(at, cutoff(V))
   I, J, Z = Int[], Int[], JMatF[]
   # a crude size hint
   for C in (I, J, Z); sizehint!(C, 24*npairs(nlist)); end
   for (i, neigs, r, R) in sites(nlist)
      nneigs = length(neigs)
      # [1] the "off-centre" component of the hessian:
      # h[a, b] = ∂_{Ra} ∂_{Rb} V     (this is a nneigs x nneigs block-matrix)
      h = hfun(V, r, R)
      for a = 1:nneigs, b = 1:nneigs
         push!(I, neigs[a])
         push!(J, neigs[b])
         push!(Z, h[a,b])
      end

      # [2] the ∂_{Ri} ∂_{Ra} terms
      # hib = ∂_{Ri} ∂_{Rb} V = - ∑_a ∂_{Ra} ∂_{Rb} V
      # also at the same time we pre-compute the centre-centre term:
      #    hii = ∂_{Ri} ∂_{Ri} V = - ∑_a ∂_{Ri} ∂_{Ra} V
      hii = zero(JMatF)
      for b = 1:nneigs
         hib = -sum( h[a, b] for a = 1:nneigs )
         hii -= hib
         append!(I, (i,         neigs[b] ))
         append!(J, (neigs[b],  i        ))
         append!(Z, (hib,       hib'     ))
      end

      # and finally add the  ∂_{Ri}^2 term, which is precomputed above
      push!(I, i)
      push!(J, i)
      push!(Z, hii)
   end
   return sparse(I, J, Z, length(at), length(at))
end






end
