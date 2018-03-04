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

write((@__FILE__()[1:end-17]) * "asedata.json", JSON.json(asedata, 0))

# NOTE:
# some other data that we could consider adding
# asedata.atomic_numbers
# ase_data.atomic_names
# ase_data.covalent_radii
# ase_data.ground_state_magnetic_moments
# ase_data.vdw_radii
#  can we get some more stuff like electron affinity somewhere?

ase_data.chemical_symbols[30]
ase_data.reference_states[30]
ase_data.reference_states[2]



function rnn_old(species::Symbol)
   X = positions(bulk(species) * 2)
   return minimum( norm(X[n]-X[m]) for n = 1:length(X) for m = n+1:length(X) )
end

_rnn = fill(-1.0, length(_symbols))
for n = 2:length(_symbols)
   z = n-1
   try
      _rnn[n] = rnn_old(JuLIP.Chemistry.chemical_symbol(z))
   catch
   end
end
