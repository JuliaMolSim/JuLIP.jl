
using NeighbourLists
using DelimitedFiles: readdlm
using JuLIP: r_sum
using LinearAlgebra: rmul!
using Requires

import JuLIP

export EAM

# ----------- An auxiliary pair potential   ϕ(r) / r

struct EAMrep{P1} <: SimplePairPotential
   f::P1 
end

evaluate!(tmp, V::EAMrep, r) = 
      evaluate!(tmp, V.f, r) / r
evaluate_d!(tmp, V::EAMrep, r) = 
      evaluate_d!(tmp, V.f, r) / r - evaluate!(tmp, V.f, r) / r^2
evaluate_dd!(tmp, V::EAMrep, r) = (
      evaluate_dd!(tmp, V.f, r) / r 
      - 2*evaluate_d!(tmp, V.f, r) / r^2
      + 2*evaluate!(tmp, V.f, r) / r^3 )

# ----------------------------------------

# Use Requires.jl to provide the ASE EAM constructor.
function __init__()
   @require ASE="51974c44-a7ed-5088-b8be-3e78c8ba416c" @eval eam_from_ase(
         filename::AbstractString; kwargs...) =
         (
            eam = ASE.Models.EAM(filename).po; # Use ASE to create calculator
            EAM(eam.nr, eam.dr, eam.nrho, eam.drho, eam.cutoff, eam.Z, eam.density_data,
                eam.embedded_data, eam.rphi_data;
                kwargs...)
         )
end

"""
   EAM{T, N, P1, P2, P3, Z} <: SitePotential

Generic N-species EAM potential.

To create the potential, one must provide three sets of functions,
the electron densities `ρ`, the embedding functions `F`, and the pair potentials `ϕ`.

Typically, the `EAM` will be created from files that provide the tabulated
functions to be interpolated.

# Constructors:
```
EAM(fname::AbstractString; kwargs...)
EAM(ϕ_file::AbstractString, ρ_file::AbstractString, F_file::AbstractString; kwargs...)
EAM(ρ::PairPotential, F::PairPotential, ϕ::PairPotential)
```

## Constructing an EAM potential from tabulated values

When using tabulated values, a few options are available.
The most flexible requires `ASE.jl` be loaded in the current session.
This provides a constructor that uses ase to read the files and provide the data.
This is the recommended option for `.fs` and `.alloy`.
`.adp` is available within ase but is not implemented here.
By default, the fits match ASE and use

Without using `ASE.jl`, a single species EAM is available which uses either a `.fs`
or three `.plt` files.

Files can e.g. be obtained from
* [NIST](https://www.ctcms.nist.gov/potentials/)

# Implementation
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
   EAM(nr::Integer, dr::Real, nrho::Integer, drho::Real, cutoff::Real,
             Z::Vector{<:Integer}, density::AbstractArray,
             embedded::AbstractMatrix, rphi::AbstractArray{<:Real,3}; kwargs...)

Constructor for the EAM using the data extracted from the parameters file by ASE.
"""
function EAM(nr::Integer, dr::Real, nrho::Integer, drho::Real, cutoff::Real,
             Z::Vector{<:Integer}, density::AbstractArray,
             embedded::AbstractMatrix, rphi::AbstractArray{<:Real,3}; kwargs...)

   r = range(0, step=dr, length=nr)
   rho = range(0, step=drho, length=nrho)

   # Copy data to ensure symmetry in the pair potential
   if size(rphi)[1] > 1
      for i=1:size(rphi)[1]
         for j=i+1:size(rphi)[2]
            rphi[j,i,:] .= rphi[i,j,:]
         end
      end
   end

   zlist = ZList(Z, sorted=false)
   ρ = allocate_array(density)
   F = allocate_array(embedded)
   rϕ = allocate_array(rphi)

   fit_splines!(rϕ, r, rphi; kwargs...)
   fit_splines!(ρ, r, density; kwargs...)
   fit_splines!(F, rho, embedded; fixcutoff=false, kwargs...) # Nonzero cutoff

   # ϕ = rϕ .* Ref(@analytic r -> 1/r)
   ϕ = EAMrep.(rϕ)

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
   EAM(ϕ_file::AbstractString, ρ_file::AbstractString,
             F_file::AbstractString; kwargs...)

Constructor for single species EAM from multiple files.
"""
function EAM(ϕ_file::AbstractString, ρ_file::AbstractString,
             F_file::AbstractString; kwargs...)
   ρ = SplinePairPotential(ρ_file; kwargs...)
   ϕ = SplinePairPotential(ϕ_file; kwargs...)
   F = SplinePairPotential(F_file; kwargs...)
   EAM(ρ, F, ϕ)
end

"""
Constructor for single species EAM.
"""
function EAM(ρ::PairPotential, F::PairPotential, ϕ::PairPotential)
   EAM(hcat(ρ), [F], hcat(ϕ), ZList([JuLIP.Chemistry.__zAny__]), max(cutoff(ϕ), cutoff(ρ)))
end

@pot EAM

cutoff(V::EAM) = V.cutoff

"""
Choose the density function, this allows for assymmetric densities.
"""
select_density_function(ρ::Matrix, i0::Integer, i::Integer) = ρ[i, i0]
select_density_function(ρ::Vector, ::Integer, i::Integer) = ρ[i]


alloc_temp(V::EAM, N::Integer, T = Float64) = 
      (  
         R = zeros(JVec{T}, N),
         Z = zeros(AtomicNumber, N),
         wrk = nothing 
      )


function evaluate!(tmp, V::EAM, Rs, Zs, z0)
   ρ = 0.0
   Es = 0.0
   i0 = z2i(V, z0)
   for (R, Z) in zip(Rs, Zs)
      i = z2i(V, Z)
      r = norm(R)
      Es += evaluate!(tmp.wrk, V.ϕ[i0,i], r) / 2
      density_function = select_density_function(V.ρ, i0, i)
      ρ += evaluate!(tmp.wrk, density_function, r)
   end
   Es += evaluate!(tmp.wrk, V.F[i0], ρ)
   return Es
end


alloc_temp_d(V::EAM, N::Integer, T = Float64) = 
      (  
         dV = zeros(JVec{T}, N),
         R = zeros(JVec{T}, N),
         Z = zeros(AtomicNumber, N),
         wrk = nothing 
      )

function evaluate_d!(dEs, tmp, V::EAM, Rs, Zs, z0)
   ρ = 0.0
   i0 = z2i(V, z0)
   for (R, Z) in zip(Rs, Zs)
      i = z2i(V, Z)
      r = norm(R)
      density_function = select_density_function(V.ρ, i0, i)
      ρ += density_function(r)
   end
   # dF = @D V.F[i0](ρ)
   dF = evaluate_d!(tmp.wrk, V.F[i0], ρ)
   for (j, (R, Z)) in enumerate(zip(Rs, Zs))
      i = z2i(V, Z)
      r = norm(R)
      # dϕ = @D V.ϕ[i0, i](r)
      dϕ = evaluate_d!(tmp.wrk, V.ϕ[i0, i], r)
      density_function = select_density_function(V.ρ, i0, i)
      # dρ = @D density_function(r)
      dρ = evaluate_d!(tmp.wrk, density_function, r)
      R̂ = R/r
      dEs[j] = (dϕ/2 + dF * dρ) * R̂
   end
   return dEs
end

evaluate_dd!(hEs, tmp, V::EAM, R, Z, z0) = _hess_!(hEs, tmp, V, R, Z, z0, identity)
precon!(hEs, tmp, V::EAM, R, Z, z0, stab=0.0) = _hess_!(hEs, tmp, V, R, Z, z0, abs, stab)

alloc_temp_dd(V::EAM, N::Integer) = 
         ( ∇ρ = zeros(JVecF, N), 
            r = zeros(Float64, N), 
            wrk = nothing 
         )

function _hess_!(hEs, tmp, V::EAM, Rs::AbstractVector{JVec{T}}, Zs, z0, fabs, stab=T(0)
                 ) where {T}
   ρ = 0.0
   i0 = z2i(V, z0)
   for (j, (R, Z)) in enumerate(zip(Rs, Zs))
      i = z2i(V, Z)
      r = norm(R)
      tmp.r[j] = r
      density_function = select_density_function(V.ρ, i0, i)
      # tmp.∇ρ[j] = @D density_function(r, R)
      tmp.∇ρ[j] = evaluate_d!(tmp.wrk, density_function, r) * (R/r)
      ρ += density_function(r)
   end
   # precompute some stuff
   # dF = @D V.F[i0](ρ)
   dF = evaluate_d!(tmp.wrk, V.F[i0], ρ)
   # ddF = @DD V.F[i0](ρ)
   ddF = evaluate_dd!(tmp.wrk, V.F[i0], ρ)
   # assemble
   for (j, (R, Z)) in enumerate(zip(Rs, Zs))
      i = z2i(V, Z)
      r = tmp.r[j]
      for k = 1:length(Rs)
         hEs[j,k] = (1-stab) * fabs(ddF) * tmp.∇ρ[j] * tmp.∇ρ[k]'
      end
      S = R / r
      density_function = select_density_function(V.ρ, i0, i)
      # dϕ = @D V.ϕ[i0,i](r)
      dϕ = evaluate_d!(tmp.wrk, V.ϕ[i0,i], r)
      # dρ = @D density_function(r)
      dρ = evaluate_d!(tmp.wrk, density_function, r)
      # ddϕ = @DD V.ϕ[i0,i](r)
      ddϕ = evaluate_dd!(tmp.wrk, V.ϕ[i0,i], r)
      # ddρ = @DD density_function(r)
      ddρ = evaluate_dd!(tmp.wrk, density_function, r)
      a = fabs(0.5 * ddϕ + dF * ddρ)
      b = fabs((0.5 * dϕ + dF *  dρ) / r)
      hEs[j,j] += ( (1-stab) * ( (a-b) * S * S' + b * I )
                   + stab  * ( (a+b) * I ) )
   end
   hEs
end


# ================= Finnis-Sinclair Potential =======================


mutable struct FSEmbed <: SimplePairPotential end
@pot FSEmbed
evaluate(::FSEmbed, ρ̄::Real) = - sqrt(ρ̄)
evaluate_d(::FSEmbed, ρ̄::T) where {T<:Real} = - T(0.5) / sqrt(ρ̄)
evaluate_dd(::FSEmbed, ρ̄::T) where {T<:Real} = T(0.25) * ρ̄^(T(-3/2))

"""
`FinnisSinclair`: constructs an EAM potential with embedding function
-√ρ̄.
"""
FinnisSinclair(pair::PairPotential, eden::PairPotential) = EAM(eden, FSEmbed(), pair)

function FinnisSinclair(fpair::AbstractString, feden::AbstractString; kwargs...)
   pair = SplinePairPotential(fpair; kwargs...)
   eden = SplinePairPotential(feden; kwargs...)
   return FinnisSinclair(pair, eden)
end


# ================= Various File Loaders =======================

function eam_from_fs(fname; kwargs...)
   F, ρfun, ϕfun, ρ, r, info = read_fs(fname)
   return EAM( [SplinePairPotential(r, ρfun; kwargs...)],
               [SplinePairPotential(ρ, F; fixcutoff=false, kwargs...)],
               [EAMrep(SplinePairPotential(r, ϕfun; kwargs...))], 
               # hcat(SplinePairPotential(r, ϕfun; kwargs...) * (@analytic r -> 1/r)),
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
