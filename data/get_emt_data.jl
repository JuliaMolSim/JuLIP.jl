
#
# Script to generate the default data-files for the EMT
# Potential; from the ASE implementation
#
# this is essentially derived from the ASE code, and hence ought to be
# licensed by whatever ASE's license is.
#

# import JuLIP
# import JuLIP.Potentials.EMT : EMTParams
using HDF5
using PyCall


# get access to the atomic numbers
@pyimport ase.calculators.emt as ase_emt
@pyimport ase.data as ase_data

# import everything we need from ASE
const Bohr = ase_emt.Bohr
# get_atomic_numbers(id::AbstractString) = ase_data.atomic_numbers[id]
const ase_parameters = ase_emt.parameters

# convert ASE style Dict to a new Dict where parameters
# are stored in a type instead of tuple.
const params = Dict{String, Dict{String,Float64}}()

beta = 1.809
maxseq = maximum([par[1] for par in values(ase_parameters)]) * Bohr
rc = beta * maxseq * 0.5 * (sqrt(3) + sqrt(4))
rr = rc * 2 * sqrt(4) / (sqrt(3) + sqrt(4))
acut = log(9999.0) / (rr - rc)


for (key, p) in ase_parameters   # p is a tuple of parameters, key a species
   s0 = p[1] * Bohr
   eta2 = p[3] / Bohr
   kappa = p[4] / Bohr
   x = eta2 * beta * s0
   gamma1 = 0.0
   gamma2 = 0.0
   for (i, n) in enumerate([12; 6; 24])
      r = s0 * beta * sqrt(i + 1.0)
      x = n / (12.0 * (1.0 + exp(acut * (r - rc))))
      gamma1 += x * exp(-eta2 * (r - beta * s0))
      gamma2 += x * exp(-kappa / beta * (r - beta * s0))
      params[key] = Dict{String, Float64}(
         "E0" => p[1], "s0" => s0, "V0" => p[2], "eta2" => eta2,
         "kappa" => kappa, "lambda" => p[5] / Bohr, "n0" => p[6] / Bohr^3,
         "rc" => rc, "gamma1" => gamma1, "gamma2" => gamma2 )
   end
end

# store: params, rc, acut, beta
# save("emt_default_params.jld", "params", params, "beta", beta, "rc", rc, "acut", acut)
h5open("emtdefaultparams.h5", "w") do file
   write(file, "params", params)
   write(file, "beta", beta)
   write(file, "rc", rc)
   write(file, "acut", acut)
end
