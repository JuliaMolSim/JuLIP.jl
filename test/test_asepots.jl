
using JuLIP.Testing, ASE, PyCall

@info("These tests to compare JuLIP vs ASE implementations of some potentials")

h3("Compare JuLIP vs ASE: EAM")
# JuLIP's EMT implementation
at = set_pbc!( bulk(:Cu, cubic=true) * (2,2,2), (true,false,false) )
rattle!(at, 0.02)
emt = EMT(at)

@info("Test JuLIP vs ASE EMT implementation")
pyemt = ASE.Models.EMTCalculator()
print("   energy: ")
println(@test abs(energy(emt, at) - energy(pyemt, at)) < 1e-10)
print("   forces: ")
println(@test norm(forces(pyemt, at) - forces(emt, at), Inf) < 1e-10)

# ------------------------------------------------------------------------


h3("Compare JuLIP EAM Implementation against ASE EAM Implementation")

@pyimport ase.calculators.eam as eam

pot_file = joinpath(dirname(pathof(JuLIP)), "..", "data", "w_eam4.fs")

@info("Generate the ASE potential")
eam4_ase = eam.EAM(potential=pot_file) |> ASECalculator

@info("Generate low-, med-, high-accuracy JuLIP potential")
eam4_jl1 = EAM(pot_file)
eam4_jl2 = EAM(pot_file; s = 1e-4)
eam4_jl3 = EAM(pot_file; s = 1e-6)

at1 = rattle!(bulk(:W, cubic=true) * 3, 0.1)
at2 = deleteat!(bulk(:W, cubic=true) * 3, 1)
at1_ase = ASEAtoms(at1)
at2_ase = ASEAtoms(at2)

for (i, (at, at_ase)) in enumerate(zip([at1, at2], [at1_ase, at2_ase]))
   @info("Test $i")
   err_low = (energy(eam4_ase, at_ase) - energy(eam4_jl1, at)) / length(at)
   err_med = (energy(eam4_ase, at_ase) - energy(eam4_jl2, at)) / length(at)
   err_hi = (energy(eam4_ase, at_ase) - energy(eam4_jl3, at)) / length(at)
   print("   Low Accuracy energy error:", err_low, "; ",
         (@test abs(err_low) < 0.02))
   print("   Low Accuracy energy error:", err_med, "; ",
         (@test abs(err_med) < 0.006))
   print("   Low Accuracy energy error:", err_hi, "; ",
         (@test abs(err_hi) < 0.0004))
end

# ------------------------------------------------------------------------
