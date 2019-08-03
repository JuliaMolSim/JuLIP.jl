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
              site_energy, site_energy_d,
              energy!, forces!, virial!,
              alloc_temp, alloc_temp_d, alloc_temp_dd

export PairPotential, SitePotential, ZeroSitePotential

# the following are prototypes for internal functions around which IPs are
# defined

function evaluate end
function evaluate_d end
function evaluate_dd end
function evaluate! end
function evaluate_d! end
function evaluate_dd! end
function grad end
function precond end


include("potentials_base.jl")
# * @pot, @D, @DD, @GRAD and related things

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
abstract type PairPotential <: SitePotential end


evaluate(V::SitePotential, R) =
      evaluate!(alloc_temp(V, length(R)), V, R)

evaluate_d(V::SitePotential, R::AbstractVector{JVec{T}}) where {T} =
      evaluate_d!(zeros(JVec{T}, length(R)),
                  alloc_temp_d(V, length(R)),
                  V, R)

evaluate_dd(V::SitePotential, R::AbstractVector{JVec{T}}) where {T} =
      evaluate_dd!(zeros(JMat{T}, length(R), length(R)),
                   alloc_temp_dd(V, length(R)),
                   V, R)

evaluate(V::PairPotential, r::Number) = evaluate!(alloc_temp(V, 1), V, r)
evaluate_d(V::PairPotential, r::Number) = evaluate_d!(alloc_temp_d(V, 1), V, r)
evaluate_dd(V::PairPotential, r::Number) = evaluate_dd!(alloc_temp_d(V, 1), V, r)

NeighbourLists.sites(at::AbstractAtoms, rcut::AbstractFloat) =
      sites(neighbourlist(at, rcut))

NeighbourLists.pairs(at::AbstractAtoms, rcut::AbstractFloat) =
      pairs(neighbourlist(at, rcut))


"a site potential that just returns zero"
mutable struct ZeroSitePotential <: SitePotential
end

@pot ZeroSitePotential

cutoff(::ZeroSitePotential) = Bool(0)
energy(V::ZeroSitePotential, at::AbstractAtoms{T}; kwargs...) where T = zero(T)
forces(V::ZeroSitePotential, at::AbstractAtoms{T}; kwargs...) where T = zeros(JVec{T}, length(at))
evaluate!(tmp, p::ZeroSitePotential, args...) = Bool(0)
evaluate_d!(dEs, tmp, V::ZeroSitePotential, args...) = fill!(dEs, zero(eltype(dEs)))
evaluate_dd!(hEs, tmp, V::ZeroSitePotential, args...) = fill!(hEs, zero(eltype(hEs)))


# Implementation of a generic site potential
# ================================================

alloc_temp(V::SitePotential, at::AbstractAtoms) =
      alloc_temp(V, maxneigs(neighbourlist(at, cutoff(V))))

alloc_temp(V::SitePotential, N::Integer) = ( R = zeros(JVecF, N), )

alloc_temp_d(V::SitePotential, at::AbstractAtoms) =
      alloc_temp_d(V, maxneigs(neighbourlist(at, cutoff(V))))

alloc_temp_d(V::SitePotential, N::Integer) = ( R = zeros(JVecF, N),
                                              dV = zeros(JVecF, N) )

alloc_temp_dd(V::SitePotential, N::Integer) = nothing

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
      _j, R = neigs!(tmp.R, nlist, i)
      E += evaluate!(tmp, V, R)
   end
   return E
end

function forces!(frc, tmp, V::SitePotential, at::AbstractAtoms{T};
                 domain=1:length(at), reset=true) where {T}
   if reset; fill!(frc, zero(JVec{T})); end
   nlist = neighbourlist(at, cutoff(V))
   for i in domain
      j, R = neigs!(tmp.R, nlist, i)
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
      _j, R = neigs!(tmp.R, nlist, i)
      evaluate_d!(tmp.dV, tmp, V, R)
      vir += site_virial(tmp.dV, R)
   end
   return vir
end


site_energies(V::SitePotential, at::AbstractAtoms{T}; kwargs...) where {T} =
      site_energies!(zeros(T, length(at)), alloc_temp(V, at), V, at)

function site_energies!(Es, tmp, V::SitePotential, at::AbstractAtoms{T};
         domain = 1:length(at)) where {T}
   nlist = neighbourlist(at, cutoff(V))
   for i in domain
      _j, R = neigs!(tmp.R, nlist, i)
      Es[i] = evaluate!(tmp, V, R)
   end
   return Es
end

site_energy(V::SitePotential, at::AbstractAtoms, i0::Integer) =
      energy(V, at; domain = (i0,))

site_energy_d(V::SitePotential, at::AbstractAtoms, i0::Integer) =
      rmul!(forces(V, at; domain = (i0,)), -one(eltype(at)))



# ------------------------------------------------
#  specialisation for Pair potentials


include("analyticpotential.jl")
# * AnalyticFunction
# * F64fun, WrappedAnalyticFunction

include("cutoffs.jl")
#   * SWCutoff
#   * ShiftCutoff
#   * SplineCutoff

include("pairpotentials.jl")
# * PairCalculator
# * LennardJonesPotential
# * MorsePotential
# * SimpleExponential
# * WrappedPairPotential

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


include("multi.jl")
# experimental multi-species code
# -> eventually this is to be integrated into all the main codebase

include("emt.jl")
# a simple analytic EAM potential. (EMT -> embedded medium theory)

end
