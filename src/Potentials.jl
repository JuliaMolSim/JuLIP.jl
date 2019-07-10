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

using JuLIP: AbstractAtoms, AbstractCalculator,
      JVec, mat, vec, JMat, SVec, vecs, SMat,
      positions, set_positions!

using StaticArrays: @SMatrix

using NeighbourLists

using LinearAlgebra: norm

using SparseArrays: sparse

import JuLIP: energy, forces, cutoff, virial, hessian_pos, hessian,
              site_energies, r_sum,
              site_energy, site_energy_d, partial_energy, partial_energy_d

export PairPotential, SitePotential, ZeroSitePotential

# the following are prototypes for internal functions around which IPs are
# defined

function evaluate end
function evaluate_d end
function evaluate_dd end
function grad end
function hess end
function precond end




"""
`PairPotential`:abstractsupertype for pair potentials

Can also be used as a constructor for analytic pair potentials, e.g.,
```julia
lj = @analytic r -> r^(-12) - 2 * r^(-6)
```
"""
abstract type PairPotential <: AbstractCalculator end

"""
`SitePotential`:abstractsupertype for generic site potentials
"""
abstract type SitePotential <: AbstractCalculator end


include("potentials_base.jl")
# * @pot, @D, @DD, @GRAD and related things

NeighbourLists.sites(at::AbstractAtoms, rcut::AbstractFloat) =
      sites(neighbourlist(at, rcut))

NeighbourLists.pairs(at::AbstractAtoms, rcut::AbstractFloat) =
      pairs(neighbourlist(at, rcut))


"a site potential that just returns zero"
mutable struct ZeroSitePotential <: SitePotential
end

@pot ZeroSitePotential


evaluate(p::ZeroSitePotential, r, R) = zero(eltype(r))
evaluate_d(p::ZeroSitePotential, r, R) = zero(r)
cutoff(::ZeroSitePotential) = Bool(0)



# Implementation of a generic site potential
# ================================================

site_energies(V::SitePotential, at::AbstractAtoms{T}) where {T} =
      T[ V(r, R) for (_₁, _₂, r, R) in sites(at, cutoff(V)) ]

energy(V::SitePotential, at::AbstractAtoms) =
      r_sum(site_energies(V, at))

evaluate(V::SitePotential, R::AbstractVector{<:JVec}) =
      evaluate(V, norm.(R), R)

evaluate_d(V::SitePotential, R::AbstractVector{<:JVec}) =
      evaluate_d(V, norm.(R), R)

function forces(V::SitePotential, at::AbstractAtoms{T}) where {T}
   frc = zeros(JVec{T}, length(at))
   for (i, j, r, R) in sites(at, cutoff(V))
      dV = @D V(r, R)
      for a = 1:length(j)
         frc[j[a]] -= dV[a]
         frc[i]    += dV[a]
      end
   end
   return frc
end

site_virial(dV, R::AbstractVector{JVec{T}}) where {T} =  (
      length(R) > 0 ? (- sum( dVi * Ri' for (dVi, Ri) in zip(dV, R) ))
                    : zero(JMat{T}) )

virial(V::SitePotential, at::AbstractAtoms) =
      sum(  site_virial((@D V(r, R)), R)
            for (_₁, _₂, r, R) in sites(at, cutoff(V))  )

function partial_energy(V::SitePotential, at::AbstractAtoms{T}, Idom) where {T}
   E = zero(T)
   nlist = neighbourlist(at, cutoff(V))
   for i in Idom
      j, r, R = neigs(nlist, i)
      E += V(r, R)
   end
   return E
end

function partial_energy_d(V::SitePotential, at::AbstractAtoms, Idom)
   F = zeros(JVec{eltype(at)}, length(at))
   nlist = neighbourlist(at, cutoff(V))
   for i in Idom
      j, r, R = neigs(nlist, i)
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
      partial_energy(V, at, (i0,))

site_energy_d(V::Union{SitePotential, PairPotential}, at::AbstractAtoms, i0::Int) =
      partial_energy_d(V, at, (i0,))


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
# code for 1-body functions ; TODO: move into `nbody`

include("hessians.jl")
# code for hessians of site potentials







end
