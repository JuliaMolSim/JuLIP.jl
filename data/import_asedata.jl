#
# ======================================================================
#   Import some chemistry and materials science data tables from
#   ASE. Without this data, JuLIP can do very little!
# ======================================================================
#
using PyCall, JSON
@pyimport ase.data as ase_data
asedata = Dict(
     :symbols => ase_data.chemical_symbols,
      :masses => ase_data.atomic_masses,
   :refstates => ase_data.reference_states
   )


write(@__DIR__() * "/asedata.json", JSON.json(asedata, 0))

# ======================================================================


# NOTE:
# some other data that we could consider adding
# asedata.atomic_numbers
# ase_data.atomic_names
# ase_data.covalent_radii
# ase_data.ground_state_magnetic_moments
# ase_data.vdw_radii
#  can we get some more stuff like electron affinity somewhere?


# function rnn_old(species::Symbol)
#    X = positions(bulk(species) * 2)
#    return minimum( norm(X[n]-X[m]) for n = 1:length(X) for m = n+1:length(X) )
# end
#
# _rnn = fill(-1.0, length(_symbols))
# for n = 2:length(_symbols)
#    z = n-1
#    try
#       _rnn[n] = rnn_old(JuLIP.Chemistry.chemical_symbol(z))
#    catch
#    end
# end


# ======================================================================


# get access to the atomic numbers
ase_emt = pyimport("ase.calculators.emt")
ase_data = pyimport("ase.data")

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

   return Dict("params" => params, "acut" => acut, "rc" => rc, "beta" => beta)
end

emt_data = emt_default_parameters()
write(@__DIR__() * "/emt.json", JSON.json(emt_data, 0))
