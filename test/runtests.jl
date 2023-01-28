
using ASE, JuLIP, Test, Printf, DataDeps
using JuLIP.Testing

h0("   JuLIP Tests   ")

@info("preparing the tests...")

verbose=true
isCI = haskey(ENV, "CI")
notCI = !isCI

## ------ loading some prototype potentials via DataDeps

test_pots = JuLIP.Deps.fetch_test_pots()

test_pots = joinpath(datadep"JuLIP_testpots", "JuLIP_data") * "/"

## ===== some prototype potentials ======
@info("Loading some interatomic potentials . .")
eam_Fe = JuLIP.Potentials.EAM(test_pots * "pfe.plt",
                              test_pots * "ffe.plt",
                              test_pots * "F_fe.plt")
print(" .")
eam_PdAgH = JuLIP.Potentials.eam_from_ase(test_pots * "PdAgH_HybridPd3Ag.eam.alloy")
print(".")
eam_W4 = JuLIP.Potentials.EAM(test_pots * "w_eam4.fs")
println(" done.")

##

julip_tests = [
   ("testaux.jl", "Miscellaneous"),
   ("test_atoms.jl", "Atoms"),
   ("test_build.jl", "Build"),
   ("test_fio.jl", "File IO"),
   ("testanalyticpotential.jl", "Analytic Potential"),
   ("testpotentials.jl", "Potentials"),
   ("test_ad.jl", "AD Potentials"),
   ("testvarcell.jl", "Variable Cell"),
   ("testhessian.jl", "Hessian"),
   ("test_onebody.jl", "One-Body"),
   ("test_eam.jl", "EAM"),
   ("test_atomsbase.jl", "AtomsBase")
   ]

# add solver tests if not on travis
if !isCI
   push!(julip_tests, ("testsolve.jl", "Solve"))
else
   @info("on CI : don't run solver tests")
end

# "testexpvarcell.jl";  # USE THIS TO WORK ON EXPCELL IMPLEMENTATION

@testset "JuLIP" begin
   for (testfile, testid) in julip_tests
      h1("Testset $(testid)")
      @testset "$(testid)" begin include(testfile); end
   end
end


# TODO:
# - fix performance regression in morseold / testanalyticpotential
# - some other analytic EAM potentials
# - finish proper EAM implementation
#   do this via ASE.jl and importing the data from ASE!!
# - maybe simplify the hessian assembly by skipping the step via
#   hessian_pos
