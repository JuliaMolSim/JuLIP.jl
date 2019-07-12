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
              site_energy, site_energy_d, partial_energy, partial_energy_d,
              energy!, forces!, virial!,
              alloc_temp, alloc_temp_d

export PairPotential, SitePotential, ZeroSitePotential

# the following are prototypes for internal functions around which IPs are
# defined

function evaluate end
function evaluate_d end
function evaluate_dd end
function grad end
function hess end
function precond end

#   Experimental Prototypes for non-allocating maps

evaluate!(       tmp,  V, args...) = evaluate(V, args...)
evaluate_d!(dEs, tmpd, V, args...) = copyto!(dEs, evaluate_d(V, args...))

include("potentials_base.jl")
# * @pot, @D, @DD, @GRAD and related things

# TODO: introduce type parameter into SitePotential{T}

"""
`SitePotential`:abstractsupertype for generic site potentials
"""
abstract type SitePotential <: AbstractCalculator end

"""
`PairPotential`:abstractsupertype for pair potentials

Can also be used as a constructor for analytic pair potentials, e.g.,
```julia
lj = @analytic r -> r^(-12) - 2 * r^(-6)
```
"""
abstract type PairPotential <: AbstractCalculator end



NeighbourLists.sites(at::AbstractAtoms, rcut::AbstractFloat) =
      sites(neighbourlist(at, rcut))

# NeighbourLists.pairs(at::AbstractAtoms, rcut::AbstractFloat) =
#       pairs(neighbourlist(at, rcut))


"a site potential that just returns zero"
mutable struct ZeroSitePotential <: SitePotential
end

@pot ZeroSitePotential

evaluate(p::ZeroSitePotential, R::AbstractVector{JVec{T}}
      ) where {T} = zero(T)
evaluate_d(p::ZeroSitePotential, R::AbstractVector{JVec{T}}
      ) where {T} = zeros(T, length(R))
cutoff(::ZeroSitePotential) = Bool(0)

evaluate!(tmp, p::ZeroSitePotential, args...) = Bool(0)
evaluate_d!(dEs, tmp, V::ZeroSitePotential, args...) = fill!(dEs, zero(eltype(dEs)))


# Implementation of a generic site potential
# ================================================

alloc_temp(V::SitePotential, at::AbstractAtoms) =
      alloc_temp(V, max_neigs(neighbourlist(at, cutoff(V))))

alloc_temp(V::SitePotential, N::Integer) = nothing

alloc_temp_d(V::SitePotential, at::AbstractAtoms) =
      alloc_temp_d(V, max_neigs(neighbourlist(at, cutoff(V))))

alloc_temp_d(V::SitePotential, N::Integer) = (dV = zeros(JVecF, N), )

energy(V::SitePotential, at::AbstractAtoms; kwargs...) =
      energy!(alloc_temp(V, at), V, at; kwargs...)

virial(V::SitePotential, at::AbstractAtoms; kwargs...) =
      virial!(alloc_temp_d(V, at), V, at; kwargs...)

forces(V::SitePotential, at::AbstractAtoms{T}; kwargs...) where {T} =
      forces!(zeros(JVec{T}, length(at)), alloc_temp_d(V, at), V, at; kwargs...)

function energy!(tmp, V::SitePotential, at::AbstractAtoms{T};
                 domain=1:length(at)) where {T}
   E = zero(T)
   nlist = neighbourlist(at, cutoff(V))
   for i in domain
      _j, _r, R = neigs(nlist, i)
      E += evaluate!(tmp, V, R)
   end
   return E
end

function forces!(frc, tmp, V::SitePotential, at::AbstractAtoms;
                 domain=1:length(at)) where {T}
   nlist = neighbourlist(at, cutoff(V))
   for i in domain
      j, _r, R = neigs(nlist, i)
      evaluate_d!(tmp.dV, tmp, V, R)
      for a = 1:length(j)
         frc[j[a]] -= tmp.dV[a]
         frc[i]    += tmp.dV[a]
      end
   end
   return frc
end

site_virial(dV, R::AbstractVector{JVec{T}}) where {T} =  (
      length(R) > 0 ? (- sum( dVi * Ri' for (dVi, Ri) in zip(dV, R) ))
                    : zero(JMat{T}) )

function virial!(tmp, V::SitePotential, at::AbstractAtoms{T};
                 domain=1:length(at)) where {T}
   vir = zero(JMat{T})
   nlist = neighbourlist(at, cutoff(V))
   for i in domain
      _j, _r, R = neigs(nlist, i)
      evaluate_d!(tmp.dV, tmp, V, R)
      vir += site_virial(tmp.dV, T)
   end
   return vir
end


site_energies(V::SitePotential, at::AbstractAtoms{T}; kwargs...) where {T} =
      site_energies!(zeros(T, length(at)), alloc_temp(V, at), V, at)

function site_energies!(Es, tmp, V::SitePotential, at::AbstractAtoms{T};
         domain = 1:length(at)) where {T}
   nlist = neighbourlist(at, cutoff(V))
   for i in domain
      _j, _r, R = neigs(nlist, i)
      Es[i] = evaluate!(tmp, V, R)
   end
   return Es
end

site_energy(V::Union{SitePotential, PairPotential}, at::AbstractAtoms, i0::Int) =
      energy(V, at; domain = (i0,))

site_energy_d(V::Union{SitePotential, PairPotential}, at::AbstractAtoms, i0::Int) =
      rmul!(forces(V, at; domain = (i0,)), -one(eltype(at)))



# ------------------------------------------------
#  specialisation for Pair potentials


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
