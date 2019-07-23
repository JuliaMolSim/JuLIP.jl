

using JuLIP.FIO: load_json
using JuLIP: chemical_symbol, atomic_number
using JuLIP.Potentials: WrappedAnalyticFunction, F64fun, @D

import JuLIP.Potentials: evaluate!, evaluate_d!


export EMT



"""
`EMT`: a re-implementation of the `EMT` calculator (a variant of EAM) os ASE
in Julia, largely for fun and comparison with Python, but also to demonstrate
how to implement a multi-component calculator in JuLIP
"""
struct EMT <: MSitePotential
   pair::Vector{WrappedPairPotential}
   Cpair::Vector{Float64}
   rho::Vector{WrappedPairPotential}
   embed::Vector{WrappedAnalyticFunction}
   z2ind::Dict{Int, Int}
   rc::Float64
end

cutoff(calc::EMT) = calc.rc + 0.5


# ========================== Initialisation  ============================

function _load_emt()
   D = load_json( joinpath(@__DIR__(), "..", "data", "emt.json") )
   return D["params"], D["acut"], D["rc"], D["beta"]
end

EMT(at::Atoms) = EMT(unique(at.Z))
EMT(sym::Vector{Symbol}) = EMT(atomic_number.(sym))

function EMT(Z::Vector{<:Integer})
   # load all the parameters
   params, acut, rc, β = _load_emt()
   rcplus = rc +  0.5   # the actual cutoff
   # get unique Z numbers and use them to allocate the EMT calculator
   Z = unique(Z)
   nZ = length(Z)
   emt = EMT(Vector{WrappedPairPotential}(undef, nZ),  # pair
             Vector{Float64}(undef, nZ),   # Cpair
             Vector{WrappedPairPotential}(undef, nZ),  # rho
             Vector{WrappedAnalyticFunction}(undef, nZ), # embed
             Dict{Int,Int}(),
             rc)
   # loop through unique symbols/Z and for each compute the relevant potentials
   for (i, z) in enumerate(Z)
      # store Z -> i maps
      emt.z2ind[z] = i
      sym = chemical_symbol(z)

      p = params[String(sym)]
      # cut-off: this is a FD function; this is AWFUL! replace with spline???
      # Θ = 1.0 / (1.0 + exp($acut * (r - $rc) ))
      # pair potential
      n0, κ, s0 = p["n0"], p["κ"], p["s0"]
      emt.pair[i] = WrappedPairPotential(
            @analytic(r -> n0 * exp( -κ * (r / β - s0) ) * θ,
                      θ = 1.0 / (1.0 + exp(acut * (r - rc) )) ), rcplus)
      emt.Cpair[i] = p["Cpair"]
      # radial electron density function
      η2, γ1 = p["η2"], p["γ1"]
      emt.rho[i] = WrappedPairPotential(
            @analytic(
            r -> n0 * exp( -η2 * (r - (β*s0)) ) * θ,
            θ = 1.0 / (1.0 + exp(acut * (r - rc) )) ), rcplus)
      # embedding function
      E0, V0, λ = p["E0"], p["V0"], p["λ"]
      emt.embed[i] = F64fun( @analytic(
         ρ̄-> E0 * ((1.0 + λ * DS) * exp(-λ * DS) - 1.0) + 6.0 * V0 * exp(-κ * DS),
         DS = - log( (1.0/(12.0*γ1*n0)) * ρ̄ ) / (β * η2) ) )
   end
   return emt
end



# ========================== Main Functionality ============================


function evaluate!(tmp, emt::EMT, 𝐑, 𝐙, z0)
   ρ̄ = 0.0
   Es = 0.0
   i0 = emt.z2ind[z0]
   for (R, Z) in zip(𝐑, 𝐙)
      i = emt.z2ind[Z]
      r = norm(R)
      Es += emt.Cpair[i0] * emt.pair[i](r)
      ρ̄ +=  emt.rho[i](r)
   end
   Es += emt.embed[i0](ρ̄)
   return Es
end

function evaluate_d!(dEs, tmp, emt::EMT, 𝐑, 𝐙, z0)
   ρ̄ = 0.0
   i0 = emt.z2ind[z0]
   for (R, Z) in zip(𝐑, 𝐙)
      i = emt.z2ind[Z]
      r = norm(R)
      ρ̄ += emt.rho[i](r)
   end
   dF = @D emt.embed[i0](ρ̄)
   for (j, (R, Z)) in enumerate(zip(𝐑, 𝐙))
      i = emt.z2ind[Z]
      r = norm(R)
      R̂ = R/r
      dpair = emt.Cpair[i0] * (@D emt.pair[i](r))
      drho = @D emt.rho[i](r)
      dEs[j] = (dpair + dF * drho) * R̂
   end
end
