"""
## module Potentials

### Summary

This module implements some basic interatomic potentials in pure Julia, as well
as provides building blocks and prototypes for further implementations
The implementation is done in such a way that they can be used either in "raw"
form or within abstract frameworks.

### Types

### `evaluate`, `evaluate_d`, `evaluate_dd`, `grad`

### The `@D`, `@DD`, `@GRAD` macros

TODO: write documentation

"""
module Potentials

using JuLIP: AbstractAtoms, AbstractNeighbourList, AbstractCalculator,
      bonds, sites,
      JVec, JVecs, mat, vec, JMat, JVecF, SVec, vecs, SMat,
      positions, set_positions!

import JuLIP: energy, forces, cutoff, virial, site_energies
import StaticArrays: @SMatrix

export Potential, PairPotential, SitePotential,
     site_energy, site_energy_d

"""
`Potential`: generic abstract supertype for all potential-like things
"""
abstract Potential <: AbstractCalculator


"""
`PairPotential`: abstract supertype for pair potentials

Can also be used as a constructor for analytic pair potentials, e.g.,
```julia
lj = PairPotential( :( r^(-12) - 2 * r^(-6) ) )
```
"""
abstract PairPotential <: Potential

"""
`SitePotential`: abstract supertype for generic site potentials
"""
abstract SitePotential <: Potential


include("potentials_base.jl")
# * @pot, @D, @DD, @GRAD and related things


# Implementation of a generic site potential
# ================================================

site_energies(pot::SitePotential, at::AbstractAtoms) =
   Float64[ pot(r, R) for (_₁, _₂, r, R, _₃) in sites(at, cutoff(pot)) ]

energy(pot::SitePotential, at::AbstractAtoms) = sum_kbn(site_energies(pot, at))

evaluate(pot::SitePotential, R::AbstractVector{JVecF}) = evaluate(pot, norm.(R), R)

evaluate_d(pot::SitePotential, R::AbstractVector{JVecF}) = evaluate_d(pot, norm.(R), R)

function forces(pot::SitePotential, at::AbstractAtoms)
   frc = zerovecs(length(at))
   for (i, j, r, R, _) in sites(at, cutoff(pot))
      dpot = @D pot(r, R)
      for a = 1:length(j)
         frc[j[a]] -= dpot[a]
      end
      frc[i] += sum(dpot)
   end
   return frc
end

site_virial(dV, R) = - sum( dVi * Ri' for (dVi, Ri) in zip(dV, R) )

virial(V::SitePotential, at::AbstractAtoms) =
      sum(  site_virial((@D V(r, R)), R)
            for (_₁, _₂, r, R, _₃) in sites(at, cutoff(V))  )



function site_energy(V::SitePotential, at::AbstractAtoms, i0::Int)
   @assert 1 <= i0 <= length(at)
   for (i, j, r, R, S) in sites(at, cutoff(V))
      if i == i0
         return V(R)
      end
   end
end

function site_energy_d(V::SitePotential, at::AbstractAtoms, i0::Int)
   @assert 1 <= i0 <= length(at)
   for (i, j, r, R, S) in sites(at, cutoff(V))
      if i == i0
         F = zeros(JVecF, length(at))
         F[j] = @D V(R)
         return F
      end
   end
end



include("analyticpotential.jl")
# * AnalyticPotential
# * AnalyticPairPotential
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

try
   include("adsite.jl")
   # * FDPotential : Site potential using ForwardDiff
   # * RDPotential : Site potential using ReverseDiffPrototype
catch
   # warn("""adsite.jl could not be included; most likely some AD package is missing;
   #    at the moment it needs `ForwardDiff, ReverseDiffPrototype`""")
end

include("EMT.jl")
# * EMTCalculator

include("stillingerweber.jl")
# * type StillingerWeber



export ZeroSitePotential

@pot type ZeroSitePotential <: Potential
end

"a site potential that just returns zero"
ZeroSitePotential

evaluate(p::ZeroSitePotential, r, R) = 0.0
evaluate_d(p::ZeroSitePotential, r, R) = zeros(r)   # TODO: is this a bug?
cutoff(::ZeroSitePotential) = 0.0



"""
`fd_hessian(V, R, h) -> H`

If `length(R) = N` and `length(R[i]) = d` then `H` is an N × N `Matrix{SMatrix}` with
each element a d × d static array.
"""
function fd_hessian{D,T}(V::SitePotential, R::Vector{SVec{D,T}}, h)
   d = length(R[1])
   N = length(R)
   H = zeros( typeof(@SMatrix zeros(d, d)), N, N )
   return fd_hessian!(H, V, R, h)
end

"""
`fd_hessian!(H, V, R, h) -> H`

Fill `H` with the hessian entries; cf `fd_hessian`.
"""
function fd_hessian!{D,T}(H, V::SitePotential, R::Vector{SVec{D,T}}, h)
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


end
