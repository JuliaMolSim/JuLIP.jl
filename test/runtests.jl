
using JuLIP, Test, Printf
using JuLIP.Testing

h0("   JuLIP Tests   ")

@info("preparing the tests...")

verbose=true

## check whether on CI
isCI = haskey(ENV, "CI")
notCI = !isCI
eam_W4 = nothing


## ===== some prototype potentials ======
@info("Loading some interatomic potentials . .")
data = joinpath(dirname(pathof(JuLIP)), "..", "data") * "/"
eam_Fe = JuLIP.Potentials.EAM(data * "pfe.plt", data * "ffe.plt", data * "F_fe.plt")
print(" .")
eam_W = JuLIP.Potentials.FinnisSinclair(data*"W-pair-Wang-2014.plt", data*"W-e-dens-Wang-2014.plt")
print(" .")
global eam_W4
try
   global eam_W4 = JuLIP.Potentials.EAM(data * "w_eam4.fs")
catch
   global eam_W4 = nothing
end
println(" done.")


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
