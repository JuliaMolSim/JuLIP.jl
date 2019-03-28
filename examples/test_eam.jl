
@info("Compare JuLIP EAM Implementation against ASE EAM Implementation")

@info("Loading libraries...")
using PyCall, JuLIP, ASE
@pyimport ase.calculators.eam as eam

pot_file = joinpath(dirname(pathof(JuLIP))[1:end-3], "data", "w_eam4.fs")

@info("Generate the ASE potential")
eam4_ase = eam.EAM(potential=pot_file) |> ASECalculator

@info("Generate low-accuracy JuLIP potential")
eam4_jl1 = EAM(pot_file)
@info("Generate medium-accuracy JuLIP potential")
eam4_jl2 = EAM(pot_file; s = 1e-4)
@info("Generate high-accuracy JuLIP potential")
eam4_jl3 = EAM(pot_file; s = 1e-6)

@info("Generate test configurations")
at1 = rattle!(bulk(:W, cubic=true) * 3, 0.1)
at2 = deleteat!(bulk(:W, cubic=true) * 3, 1)
at1_ase = ASEAtoms(at1)
at2_ase = ASEAtoms(at2)

@info("Test 1")
println("   Low Accuracy energy error:",
      (energy(eam4_ase, at1_ase) - energy(eam4_jl1, at1)) / length(at1))
println("Medium Accuracy energy error:",
      (energy(eam4_ase, at1_ase) - energy(eam4_jl2, at1)) / length(at1))
println("  High Accuracy energy error:",
      (energy(eam4_ase, at1_ase) - energy(eam4_jl3, at1)) / length(at1))

@info("Test 2")
println("   Low Accuracy energy error:",
      (energy(eam4_ase, at2_ase) - energy(eam4_jl1, at2)) / length(at2))
println("Medium Accuracy energy error:",
      (energy(eam4_ase, at2_ase) - energy(eam4_jl2, at2)) / length(at2))
println("  High Accuracy energy error:",
      (energy(eam4_ase, at2_ase) - energy(eam4_jl3, at2)) / length(at2))
