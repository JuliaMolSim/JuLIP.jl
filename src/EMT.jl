#
# Port of ASE's EMT calculator to pure Julia
# (only the parameters are imported from ASE)
#
#
# TODO: make it FAST FAST FAST (competitive with ASAP)
#


using JuLIP: AbstractCalculator, neighbourlist
import JuLIP.Potentials: cutsw, cutsw_d, evaluate, evaluate_d, HS
using JuLIP.ASE: ASEAtoms, chemical_symbols
using PyCall


# get access to the atomic numbers
@pyimport ase.calculators.emt as ase_emt
@pyimport ase.data as ase_data


"""
`EMT`: a re-implementation of the `EMT` calculator (a variant of EAM) os ASE
in Julia, largely for fun and comparison with Python; the goal is to make this
so efficient that it can compete with ASAP.
"""
type EMTCalculator <: Potential
   pair::Vector{HS{WrappedAnalyticFunction}}
   Cpair::Vector{Float64}
   rho::Vector{HS{WrappedAnalyticFunction}}
   embed::Vector{WrappedAnalyticFunction}
   ind::Vector{Int}
   rc::Float64
end

EMTCalculator(n::Int) = EMTCalculator(
         Vector{HS{WrappedAnalyticFunction}}(n), Vector{Float64}(n),   # pair, Cpair
         Vector{HS{WrappedAnalyticFunction}}(n),
         Vector{WrappedAnalyticFunction}(n), Vector{Int}(0), 0.0 )

cutoff(calc::EMTCalculator) = calc.rc + 0.5

# ========================== Load the Parameter Set ============================

function emt_default_parameters()
   # import everything we need from ASE
   Bohr = ase_emt.Bohr
   # get_atomic_numbers(id::AbstractString) = ase_data.atomic_numbers[id]
   ase_parameters = ase_emt.parameters
   beta = ase_emt.beta
   # convert ASE style Dict to a new Dict where parameters
   # are stored in a type instead of tuple.
   params = Dict{String, Dict{String,Float64}}()
   maxseq = maximum([par[2] for par in values(ase_parameters)]) * Bohr  # ✓
   rc = beta * maxseq * 0.5 * (sqrt(3) + sqrt(4))   # ✓
   rr = rc * 2 * sqrt(4) / (sqrt(3) + sqrt(4))   # ✓
   acut = log(9999.0) / (rr - rc)      # ✓

   for (key, p) in ase_parameters   # p is a tuple of parameters, key a species
      s0 = p[2] * Bohr
      eta2 = p[4] / Bohr
      kappa = p[5] / Bohr
      x = eta2 * beta * s0
      gamma1 = 0.0
      gamma2 = 0.0
      for (i, n) in enumerate([12; 6; 24])
         r = s0 * beta * sqrt(i)
         x = n / (12.0 * (1.0 + exp(acut * (r - rc))))
         gamma1 += x * exp(-eta2 * (r - beta * s0))
         gamma2 += x * exp(-kappa / beta * (r - beta * s0))
      end
      params[key] = Dict{String, Float64}(
                     "E0" => p[1], "s0" => s0, "V0" => p[3], "η2" => eta2,
                     "κ" => kappa, "λ" => p[6] / Bohr, "n0" => p[7] / Bohr^3,
                     "γ1" => gamma1, "γ2" => gamma2 )
      par = params[key]
      par["Cpair"] = -0.5 * par["V0"] / par["γ2"] / par["n0"]
   end

   return params, acut, rc, beta
end

# ========================== Initialisation  ============================

EMTCalculator(at::ASEAtoms) =
   init!( EMTCalculator(length(unique(chemical_symbols(at)))), at )

function init!(calc::EMTCalculator, at::ASEAtoms)
   # load all the parameters
   params, acut, rc, β = emt_default_parameters()
   calc.rc = rc
   rcplus = cutoff(calc)
   # get a list of all chemical symbols existing in this atoms object
   #  (don't worry about computing them twice)
   symbols = chemical_symbols(at)
   # loop through unique symbols and for each compute the relevant potentials
   sym_to_ind = Dict{typeof(symbols[1]), Int}()
   for (i, s) in enumerate(unique(symbols))
      sym_to_ind[s] = i
      p = params[s]
      # cut-off: this is a FD function; this is AWFUL! replace with spline???
      # Θ = 1.0 / (1.0 + exp($acut * (r - $rc) ))
      # pair potential
      n0, κ, s0 = p["n0"], p["κ"], p["s0"]
      calc.pair[i] = F64fun( @analytic(
            r -> n0 * exp( -κ * (r / β - s0) ) * θ,
            θ = 1.0 / (1.0 + exp(acut * (r - rc) )) ) ) * HS(rcplus)
      calc.Cpair[i] = p["Cpair"]
      # radial electron density function
      η2, γ1 = p["η2"], p["γ1"]
      calc.rho[i] = F64fun( @analytic(
            r -> n0 * exp( -η2 * (r - (β*s0)) ) * θ,
            θ = 1.0 / (1.0 + exp(acut * (r - rc) )) ) ) * HS(rcplus)
      # embedding function
      Crho = 1.0 / γ1 / n0    # TODO: get rid of Crho
      E0, V0, λ = p["E0"], p["V0"], p["λ"]
      calc.embed[i] = F64fun( @analytic(
         ρ̄-> E0 * ((1.0 + λ * DS) * exp(-λ * DS) - 1.0) + 6.0 * V0 * exp(-κ * DS),
         DS = - log( (Crho/12.0) * ρ̄ ) / (β * η2) ) )
   end
   # for each atom, determine which index in the potential arrays it
   # corresponds to
   calc.ind = zeros(Int, length(at))
   for n = 1:length(at)
      calc.ind[n] = sym_to_ind[symbols[n]]
   end
   return calc
end



# ========================== Main Functionality ============================

function energy(calc::EMTCalculator, at::ASEAtoms)
   E = 0.0
   ρ̄ = zeros(length(at))   # store the array of electron densities
   for (i,j,r,_,_) in bonds(at, cutoff(calc))
      si, sj = calc.ind[i], calc.ind[j]
      E += calc.Cpair[si] * calc.pair[sj](r)
      ρ̄[i] += calc.rho[sj](r)
   end
   for n = 1:length(at)
      sn = calc.ind[n]
      E += calc.embed[sn](ρ̄[n])
   end
   return E
end


function forces(calc::EMTCalculator, at::ASEAtoms)
   dE = zerovecs(length(at))  # allocate force vector
   ρ̄ = zeros(length(at))      # store the array of electron densities
   dρ̄ = zerovecs(length(at))  #   ... + derivatives
   nlist = neighbourlist(at, cutoff(calc))
   for (i,j,r,R,_) in bonds(nlist)
      si, sj = calc.ind[i], calc.ind[j]
      # E += calc.Cpair[si] * calc.pair[sj](r)
      dV = (calc.Cpair[si] * (@D calc.pair[sj](r)) / r) * R
      dE[j] -= dV
      dE[i] += dV
      ρ̄[i] += calc.rho[sj](r)
   end
   # compute the derivatives of the embedding function
   dF = zeros(length(at))
   for n = 1:length(at)
      sn = calc.ind[n]
      dF[n] = @D calc.embed[sn](ρ̄[n])
   end
   # compute the resulting forces from the embedding terms
   for (i,j,r,R,_) in bonds(nlist)
      # ρ̄[i] += calc.rho[sj](r)
      # E += calc.embed[sn](ρ̄[n])
      sj = calc.ind[j]
      dρ = ((@D calc.rho[sj](r))/r) * R
      dE[j] -= dF[i] * dρ
      dE[i] += dF[i] * dρ
   end
   return dE
end
