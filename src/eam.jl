export EAM

using NeighbourLists
using DelimitedFiles: readdlm
using JuLIP: r_sum

using LinearAlgebra: rmul!

# =================== General Single-Species EAM Potential ====================
# TODO: Alloy potential

"""
`struct EAM1`

Generic Single-species EAM potential, to specify it, one needs to
specify the pair potential ϕ, the electron density ρ and the embedding
function F.

The most convenient constructor is likely via tabulated values,
more below.

# Constructors:
```
EAM(pair::PairPotential, eden::PairPotential, embed)
EAM(fpair::AbstractString, feden::AbstractString, fembed::AbstractString; kwargs...)
```

## Constructing an EAM potential from tabulated values

At the moment only the .plt format is implemented. Files can e.g. be
obtained from
* [NIST](https://www.ctcms.nist.gov/potentials/)

Use the `EAM(fpair, feden, fembed)` constructure. The keyword arguments specify
details of how the tabulated values are fitted; see
`?SplinePairPotential` for more details.

TODO: implement other file formats.
"""
struct EAM1{T1, T2, T3, T4} <: SitePotential
   ϕ::T1    # pair potential
   ρ::T2    # electron density potential
   F::T3    # embedding function
   info::T4
end

@pot EAM1



EAM1(ϕ, ρ, F) = EAM1(ϕ, ρ, F, nothing)

cutoff(V::EAM1) = max(cutoff(V.ϕ), cutoff(V.ρ))

# evaluate(V::EAM1, r, R) =
#     length(r) == 0 ? V.F(0.0) :
#            V.F( sum(t->V.ρ(t), r) ) + 0.5 * sum(t->V.ϕ(t), r)
evaluate(V::TV, r, R) where TV <: EAM1 =
    length(r) == 0 ? V.F(0.0) :
           V.F( sum(t->V.ρ(t), r) ) + 0.5 * sum(t->V.ϕ(t), r)

# TODO: this creates a lot of unnecessary overhead; probaby better to
#       define vectorised versions of pair potentials
function evaluate_d(V::EAM1, r, R)
   if length(r) == 0; return JVecF[]; end
   ρ̄ = sum(V.ρ(s) for s in r)
   dF = @D V.F(ρ̄)
   #         (0.5 * ϕ'          + F' *  ρ')           * ∇r     (where ∇r = R/r)
   return [ ((0.5 * (@D V.ϕ(s)) + dF * (@D V.ρ(s))) / s) * S  for (s, S) in zip(r, R) ]
end


function evaluate_d!(out, V::EAM1, rs, Rs)
   if length(rs) == 0; return out; end
   ρ̄ = sum(V.ρ(s) for s in rs)
   dF = @D V.F(ρ̄)
   for (i, (r, R)) in enumerate(zip(rs, Rs))
      out[i] += ((0.5 * (@D V.ϕ(r)) + dF * (@D V.ρ(r))) / r) * R
   end
   return out
end


# TODO: which of the two `evaluate_dd` and `hess` should we be using?
#       probably these two should be equivalent
evaluate_dd(V::EAM1, r, R) = hess(V, r, R)
hess(V::EAM1, r, R) = _hess_(V, r, R, identity)

# ff preconditioner specification for EAM potentials
#   (just replace id with abs and hess with precon in the hessian code)
precon(V::EAM1, r, R, stab=0.0) = _hess_(V, r, R, abs, stab)


function _hess_(V::EAM1, r, R, fabs, stab=0.0)
   # allocate storage
   H = zeros(JMatF, length(r), length(r))
   # precompute some stuff
   ρ̄ = sum( V.ρ(s, S)  for (s, S) in zip(r, R) )
   ∇ρ = [ grad(V.ρ, s, S) for (s, S) in zip(r, R) ]
   dF = @D V.F(ρ̄)
   ddF = @DD V.F(ρ̄)
   # something to stabilize for the precon version
   Id = one(JMatF)
   # assemble
   for i = 1:length(r)
      for j = 1:length(r)
         H[i,j] = (1-stab) * fabs(ddF) * ∇ρ[i] * ∇ρ[j]'
      end
      S = R[i] / r[i]
      dϕ = @D V.ϕ(r[i])
      dρ = @D V.ρ(r[i])
      ddϕ = @DD V.ϕ(r[i])
      ddρ = @DD V.ρ(r[i])
      a = fabs(0.5 * (ddϕ + dF * ddρ))
      b = fabs((0.5 * (dϕ) + dF * (dρ)) / r[i])
      H[i,i] += ( (1-stab) * ( a * S * S' + b * (Id - S * S') )
                   + stab  * ( (a+b) * Id ) )
   end
   return H
end


# implementation of EAM models using data files

function EAM(fpair::AbstractString, feden::AbstractString,
             fembed::AbstractString; kwargs...)
   pair = SplinePairPotential(fpair; kwargs...)
   eden = SplinePairPotential(feden; kwargs...)
   embed = SplinePairPotential(feden; fixcutoff = false, kwargs...)
   return EAM1(pair, eden, embed)
end

#
# Load EAM file from .fs file format
#
function EAM(fname::AbstractString; kwargs...)

   if fname[end-3:end] == ".eam"
      error(".eam is not yet implemented, please file an issue")
   elseif fname[end-6:end] == ".eam.fs"
      error(".eam.fs is not yet implemented, please file an issue")
   elseif fname[end-2:end] == ".fs"
      return eam_from_fs(fname; kwargs...)
   end

   error("unknwon EAM file format, please file an issue")
end


# ================= Finnis-Sinclair Potential =======================


mutable struct FSEmbed end
@pot FSEmbed
evaluate(V::FSEmbed, ρ̄) = - sqrt(ρ̄)
evaluate_d(V::FSEmbed, ρ̄) = - 0.5 / sqrt(ρ̄)
evaluate_dd(V::FSEmbed, ρ̄) = 0.25 * ρ̄^(-3/2)

"""
`FinnisSinclair`: constructs an EAM potential with embedding function
-√ρ̄.
"""
FinnisSinclair(pair::PairPotential, eden::PairPotential) =
   EAM1(pair, eden, FSEmbed())

function FinnisSinclair(fpair::AbstractString, feden::AbstractString; kwargs...)
   pair = SplinePairPotential(fpair; kwargs...)
   eden = SplinePairPotential(feden; kwargs...)
   return FinnisSinclair(pair, eden)
end


# ================= Various File Loaders =======================

"""
`eam_from_fs(fname; kwargs...) -> EAM`

Read a `.fs` file specifying and EAM / Finnis-Sinclair potential.
"""
function eam_from_fs(fname; kwargs...)
   F, ρfun, ϕfun, ρ, r, info = read_fs(fname)
   return EAM1( SplinePairPotential(r, ϕfun; kwargs...) * (@analytic r -> 1/r),
               SplinePairPotential(r, ρfun; kwargs...),
               SplinePairPotential(ρ, F; fixcutoff= false, kwargs...),
               info )
end


"""
`read_fs(fname)` -> F, ρfun, ϕfun, ρ, r, info

Read a `.fs` file specifying and EAM / Finnis-Sinclair potential.
The description of the file format is taken from
   http://lammps.sandia.gov/doc/pair_eam.html
see also
   https://sites.google.com/a/ncsu.edu/cjobrien/tutorials-and-guides/eam
"""
function read_fs(fname)
   f = open(fname)
   # ignore the first three lines
   for n = 1:3
      readline(f)
   end

   # line 4: Nelements Element1 Element2 ... ElementN
   L4 = readline(f) |> chomp |> IOBuffer |> readdlm
   if L4[1] != 1
      error("""`read_fs`: the file `$fname` is for alloys, but only
               the single species potential is implemented so far.""")
   end
   info = Dict()
   info[:species] = L4[2]

   # line 5: Nrho, drho, Nr, dr, cutoff
   L5 = readline(f) |> IOBuffer |> readdlm
   Nrho, drho, Nr, dr, cutoff = Int(L5[1]), L5[2], Int(L5[3]), L5[4], L5[5]
   info[:cutoff] = cutoff

   # line 6: atomic number, mass, lattice constant, lattice type (e.g. FCC)
   L6 = readline(f) |> IOBuffer |> readdlm
   info[:number], info[:mass], info[:a0], info[:lattice] = tuple(L6...)

   # all the data
   data = readdlm(f)
   @assert length(data) == Nrho+2*Nr
   ρ = range(0, stop=(Nrho-1)*drho, length=Nrho)
   r = range(cutoff - (Nr-1)*dr, stop=cutoff, length=Nr)

   # embedding function
   F = data[1:Nrho]
   # density function
   ρfun  = data[Nrho+1:Nrho+Nr]
   # interatomic potential
   ϕfun  = data[Nrho+Nr+1:Nrho+2*Nr]

   return F, ρfun, ϕfun, ρ, r, info
end




# ================= Efficient implementation of EAM forces =======================

import JuLIP: energy, forces

# the main justification for these codes is that vectorised evaluation of the
# splines gives a factor 2 speed-up, probably this is primarily due to
# the function call overhead (maybe also allocations)
# In fact, this optimised codes gives a factor 3 speed-up for energy and factor 7
# for forces
# TODO: need a proper julia spline library

function _rhobar(V::EAM1, nlist::PairList)
   ρ = V.ρ(nlist.r)
   ρ̄ = zeros(nsites(nlist))
   for n = 1:npairs(nlist)
      ρ̄[nlist.i[n]] += ρ[n]
   end
   return ρ̄
end

function energy(V::EAM1{SplinePairPotential, SplinePairPotential, T},
                        at::AbstractAtoms) where T
   nlist = neighbourlist(at, cutoff(V))::PairList
   ρ̄ = _rhobar(V, nlist)
   ϕ = V.ϕ(nlist.r)
   return sum( V.F(s) for s in ρ̄ ) + 0.5 * sum(ϕ)
end

function forces(V::EAM1{SplinePairPotential, SplinePairPotential, T},
                        at::AbstractAtoms) where T
   nlist = neighbourlist(at, cutoff(V))::PairList
   ρ̄ = _rhobar(V, nlist)
   dF = [ @D V.F(t)  for t in ρ̄ ]
   dρ = @D V.ρ(nlist.r)
   dϕ = @D V.ϕ(nlist.r)

   # compute the forces
   dE = zerovecs(length(at))
   for n = 1:npairs(nlist)
      f = ((0.5 * dϕ[n] + dF[nlist.i[n]] * dρ[n])/nlist.r[n]) * nlist.R[n]
      dE[nlist.i[n]] += f
      dE[nlist.j[n]] -= f
   end
   return dE
end


export energy_map, forces_map
using JuLIP:Atoms

energy_map(V::EAM1, at::Atoms) =
   r_sum( maptosites!( (r,R) -> V(r,R),
                         zeros(length(at)),
                         sites(at, cutoff(V)) ) )

forces_map(V::EAM1, at::Atoms) =
   rmul!( maptosites_d!( ((r, R) -> @D V(r, R)),
                          zeros(JVecF, length(at)),
                          sites(at, cutoff(V)) ), -1 )
