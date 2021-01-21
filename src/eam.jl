
using NeighbourLists
using DelimitedFiles: readdlm
using JuLIP: r_sum
using LinearAlgebra: rmul!
using Requires

import JuLIP

export EAM
export EAM1

# Use Requires.jl to provide the ASE EAM constructor.
function __init__()
   @require ASE="51974c44-a7ed-5088-b8be-3e78c8ba416c" @eval eam_from_ase(
         filename::AbstractString; kwargs...) = 
         (
            eam = ASE.Models.EAM(filename).po; # Use ASE to create calculator
            EAM(eam.r, eam.rho, eam.Z, eam.density_data,
                eam.embedded_data, eam.rphi_data, eam.cutoff)
         )
end

"""
   EAM{T<:Real, P<:SimplePairPotential, Z<:AbstractZList} <: SitePotential

EAM potential for multiple species.

`ρ` can be a `Vector`  or a `Matrix`, depending on the symmetry of the density function.
`.fs` files can provide an assymmetric density.
"""
struct EAM{T<:Real, N,
           P1<:SimplePairPotential, P2<:SimplePairPotential, P3<:SimplePairPotential,
           Z<:AbstractZList} <: SitePotential
   ρ::Array{P1, N}
   F::Vector{P2}
   ϕ::Matrix{P3}
   zlist::Z
   cutoff::T
end

"""
   EAM(fname::AbstractString; kwargs...)

Load EAM from a single parameter file.

If `ASE.jl` is loaded, it will use ASE to parse the file. This should work for `eam.alloy`,
`eam.fs` and `.fs`. Still needs compatibility for `.adp`.

Otherwise, it will attempt to read from a single species `.fs` file.
"""
function EAM(fname::AbstractString; kwargs...)

   try
      return eam_from_ase(fname; kwargs...) # Provided if `ASE.jl` is loaded
   catch e
      if fname[end-2:end] == ".fs" && fname[end-6:end] != ".eam.fs"
         return eam_from_fs(fname; kwargs...)
      else
         print(e)
         error("Not an `.fs` file, if using a different EAM format, try `using ASE.jl`.")
      end
   end
end

"""
   EAM(r::Vector, rho::Vector, Z::Vector{<:Integer}, density::Matrix,
       embedded::Matrix, rphi::Array{T,3}, cutoff::T; kwargs...) where {T}

Constructor for the EAM using the data extracted from the parameters file by ASE.
"""
function EAM(r::AbstractVector, rho::AbstractVector, Z::Vector{<:Integer}, density::AbstractArray,
             embedded::Matrix, rphi::Array{T,3}, cutoff::T; kwargs...) where {T}

   # Copy data to ensure symmetry in the pair potential
   if size(rphi)[1] > 1
      for i=1:size(rphi)[1]
         for j=i+1:size(rphi)[2]
            rphi[j,i,:] .= rphi[i,j,:]
         end
      end
   end

   zlist = ZList(Z)
   ρ = allocate_array(density)
   F = allocate_array(embedded)
   rϕ = allocate_array(rphi)

   fit_splines!(rϕ, r, rphi; kwargs...)
   fit_splines!(ρ, r, density; kwargs...) 
   fit_splines!(F, rho, embedded; fixcutoff=false, kwargs...) # Nonzero cutoff

   ϕ = rϕ .* Ref(@analytic r -> 1/r)

   EAM(ρ, F, ϕ, zlist, cutoff)
end

"""
Allocate array for storing spline potentials
"""
allocate_array(data::AbstractArray) = Array{SplinePairPotential}(undef, size(data)[1:end-1])

"""
   fit_splines!(out, x, y; kwargs...)

Fit the data along the final dimension of `y` and store in `out`.
"""
function fit_splines!(out, x, y; kwargs...)
   for I in CartesianIndices(out)
      out[I] = SplinePairPotential(x, y[Tuple(I)...,:]; kwargs...)
   end
end

"""
   EAM(fpair::AbstractString, feden::AbstractString,
       fembed::AbstractString; kwargs...)

Constructor for single species from multiple files.

Provides same energies and forces as existing EAM1.
"""
function EAM(fpair::AbstractString, feden::AbstractString,
             fembed::AbstractString; kwargs...)
   ρ = SplinePairPotential(feden; kwargs...)
   ϕ = SplinePairPotential(fpair; kwargs...)
   F = SplinePairPotential(fembed; fixcutoff = false, kwargs...)
   EAM([ρ], [F], hcat(ϕ), ZList([JuLIP.Chemistry.__zAny__]), max(cutoff(ϕ), cutoff(ρ)))
end

@pot EAM

cutoff(V::EAM) = V.cutoff

"""
Choose the density function, this allows for assymmetric densities.
"""
select_density_function(ρ::Matrix, i0::Integer, i::Integer) = ρ[i, i0]
select_density_function(ρ::Vector, ::Integer, i::Integer) = ρ[i]

function evaluate!(tmp, V::EAM, Rs, Zs, z0)
   ρ = 0.0
   Es = 0.0
   i0 = z2i(V, z0)
   for (R, Z) in zip(Rs, Zs)
      i = z2i(V, Z)
      r = norm(R)
      Es += V.ϕ[i0,i](r)/2
      density_function = select_density_function(V.ρ, i0, i)
      ρ += density_function(r)
   end
   Es += V.F[i0](ρ)
   Es
end

function evaluate_d!(dEs, tmp, V::EAM, Rs, Zs, z0)
   ρ = 0.0
   i0 = z2i(V, z0)
   for (R, Z) in zip(Rs, Zs)
      i = z2i(V, Z)
      r = norm(R)
      density_function = select_density_function(V.ρ, i0, i)
      ρ += density_function(r)
   end
   dF = @D V.F[i0](ρ)
   for (j, (R, Z)) in enumerate(zip(Rs, Zs))
      i = z2i(V, Z)
      r = norm(R)
      dϕ = @D V.ϕ[i0, i](r)
      density_function = select_density_function(V.ρ, i0, i)
      dρ = @D density_function(r)
      R̂ = R/r
      dEs[j] = (dϕ/2 + dF * dρ) * R̂
   end
   dEs
end

evaluate_dd!(hEs, tmp, V::EAM, R, Z, z0) = _hess_!(hEs, tmp, V, R, Z, z0, identity)
precon!(hEs, tmp, V::EAM, R, Z, z0, stab=0.0) = _hess_!(hEs, tmp, V, R, Z, z0, abs, stab)
alloc_temp_dd(::EAM, N::Integer) = (∇ρ = zeros(JVecF, N), r = zeros(Float64, N))

function hessian(calc::EAM, at::AbstractAtoms)
   if JuLIP.fixedcell(at)
      H =  ad_hessian(calc, at)
      return JuLIP.projectxfree(at, H)
   end
   @error("`hessian` is not yet implemented for variable cells")
   return nothing
end

function _hess_!(hEs, tmp, V::EAM, Rs::AbstractVector{JVec{T}}, Zs, z0, fabs, stab=T(0)
                 ) where {T}
   ρ = 0.0
   i0 = z2i(V, z0)
   for (j, (R, Z)) in enumerate(zip(Rs, Zs))
      i = z2i(V, Z)
      r = norm(R)
      tmp.r[j] = r
      density_function = select_density_function(V.ρ, i0, i)
      tmp.∇ρ[j] = @D density_function(r, R)
      ρ += density_function(r)
   end
   # precompute some stuff
   dF = @D V.F[i0](ρ)
   ddF = @DD V.F[i0](ρ)
   # assemble
   for (j, (R, Z)) in enumerate(zip(Rs, Zs))
      i = z2i(V, Z)
      r = tmp.r[j]
      for k = 1:length(Rs)
         hEs[j,k] = (1-stab) * fabs(ddF) * tmp.∇ρ[j] * tmp.∇ρ[k]'
      end
      S = R / r
      density_function = select_density_function(V.ρ, i0, i)
      dϕ = @D V.ϕ[i0,i](r)
      dρ = @D density_function(r)
      ddϕ = @DD V.ϕ[i0,i](r)
      ddρ = @DD density_function(r)
      a = fabs(0.5 * ddϕ + dF * ddρ)
      b = fabs((0.5 * dϕ + dF *  dρ) / r)
      hEs[j,j] += ( (1-stab) * ( (a-b) * S * S' + b * I )
                   + stab  * ( (a+b) * I ) )
   end
   hEs
end

# =================== General Single-Species EAM Potential ====================

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
struct EAM1{T1, T2, T3, T4} <: SimpleSitePotential
   ϕ::T1    # pair potential
   ρ::T2    # electron density potential
   F::T3    # embedding function
   info::T4
end

function EAM1(eam::EAM)
   @assert length(eam.zlist) == 1 "This EAM is defined for more than 1 species"
   EAM1(eam.ϕ[1], eam.ρ[1], eam.F[1])
end

function EAM1(fpair::AbstractString, feden::AbstractString,
             fembed::AbstractString; kwargs...)
   pair = SplinePairPotential(fpair; kwargs...)
   eden = SplinePairPotential(feden; kwargs...)
   embed = SplinePairPotential(fembed; fixcutoff = false, kwargs...)
   return EAM1(pair, eden, embed)
end

@pot EAM1


EAM1(ϕ, ρ, F) = EAM1(ϕ, ρ, F, nothing)

cutoff(V::EAM1) = max(cutoff(V.ϕ), cutoff(V.ρ))

function evaluate!(tmp, V::EAM1, R::AbstractVector{JVec{T}}) where {T}
   if  length(R) == 0
      return V.F(T(0.0))
   else
      return V.F(sum(V.ρ ∘ norm, R)) + T(0.5) * sum(V.ϕ ∘ norm, R)
   end
end

function evaluate_d!(dEs::AbstractVector{JVec{T}}, tmp, V::EAM1, Rs) where {T}
   if length(Rs) == 0; return dEs; end
   ρ̄ = sum(V.ρ ∘ norm, Rs)
   dF = @D V.F(ρ̄)
   for (i, R) in enumerate(Rs)
      r = norm(R)
      dEs[i] = ((T(0.5) * (@D V.ϕ(r)) + dF * (@D V.ρ(r))) / r) * R
   end
   return dEs
end


alloc_temp_dd(V::EAM1, N::Integer) =
      ( ∇ρ = zeros(JVecF, N),
         r = zeros(Float64, N) )

evaluate_dd!(hEs, tmp, V::EAM1, R) = _hess_!(hEs, tmp, V, R, identity)

# ff preconditioner specification for EAM potentials
#   (just replace id with abs and hess with precon in the hessian code)
precon!(hEs, tmp, V::EAM1, R, stab=0.0) = _hess_!(hEs, tmp, V, R, abs, stab)



function hessian(calc::EAM1, at::AbstractAtoms)
   if JuLIP.fixedcell(at)
      H =  ad_hessian(calc, at)
      return JuLIP.projectxfree(at, H)
   end
   @error("`hessian` is not yet implemented for variable cells")
   return nothing
end



function _hess_!(hEs, tmp, V::EAM1, R::AbstractVector{JVec{T}}, fabs, stab=T(0)
                 ) where {T}
   for i = 1:length(R)
      r = norm(R[i])
      tmp.r[i] = r
      tmp.∇ρ[i] = @D V.ρ(r, R[i])
   end
   # precompute some stuff
   ρ̄ = sum(V.ρ, tmp.r)
   dF = @D V.F(ρ̄)
   ddF = @DD V.F(ρ̄)
   # assemble
   for i = 1:length(R)
      for j = 1:length(R)
         hEs[i,j] = (1-stab) * fabs(ddF) * tmp.∇ρ[i] * tmp.∇ρ[j]'
      end
      r = tmp.r[i]
      S = R[i] / r
      dϕ = @D V.ϕ(r)
      dρ = @D V.ρ(r)
      ddϕ = @DD V.ϕ(r)
      ddρ = @DD V.ρ(r)
      a = fabs(0.5 * ddϕ + dF * ddρ)
      b = fabs((0.5 * dϕ + dF *  dρ) / r)
      hEs[i,i] += ( (1-stab) * ( (a-b) * S * S' + b * I )
                   + stab  * ( (a+b) * I ) )
   end
   return hEs
end


# ================= Finnis-Sinclair Potential =======================


mutable struct FSEmbed end
@pot FSEmbed
evaluate(V::FSEmbed, ρ̄) = - sqrt(ρ̄)
evaluate_d(V::FSEmbed, ρ̄::T) where {T} = - T(0.5) / sqrt(ρ̄)
evaluate_dd(V::FSEmbed, ρ̄::T) where {T} = T(0.25) * ρ̄^(T(-3/2))

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
function eam1_from_fs(fname; kwargs...)
   F, ρfun, ϕfun, ρ, r, info = read_fs(fname)
   return EAM1( SplinePairPotential(r, ϕfun; kwargs...) * (@analytic r -> 1/r),
               SplinePairPotential(r, ρfun; kwargs...),
               SplinePairPotential(ρ, F; fixcutoff= false, kwargs...),
               info )
end

function eam_from_fs(fname; kwargs...)
   F, ρfun, ϕfun, ρ, r, info = read_fs(fname)
   return EAM( [SplinePairPotential(r, ρfun; kwargs...)],
               [SplinePairPotential(ρ, F; fixcutoff=false, kwargs...)],
               hcat(SplinePairPotential(r, ϕfun; kwargs...) * (@analytic r -> 1/r)),
               ZList(Symbol.(info[:species])),
               info[:cutoff])
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
