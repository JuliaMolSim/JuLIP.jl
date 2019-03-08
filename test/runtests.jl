
using JuLIP, Test, Printf
using JuLIP.Testing

include("aux.jl")
verbose=true

## check whether on CI
isCI = haskey(ENV, "CI")
notCI = !isCI
eam_W4 = nothing

## check whether ASE is available
global hasase = true
try
   import ASE
catch
   global hasase = false
end

julip_tests = [
   ("testaux.jl", "Miscellaneous"),
   ("test_atoms.jl", "Atoms"),
   ("test_build.jl", "Build"),
   ("testanalyticpotential.jl", "Analytic Potential"),
   ("testpotentials.jl", "Potentials"),
   ("test_ad.jl", "AD Potentials"),
   ("testvarcell.jl", "Variable Cell"),
   ("testhessian.jl", "Hessian"),
   ("testsolve.jl", "Solve"),
   ("test_fio.jl", "File IO"),
]

# remove testsolve if on Travis
if isCI
   julip_tests = julip_tests[1:end-1]
end

# "testexpvarcell.jl";  # USE THIS TO WORK ON EXPCELL IMPLEMENTATION

## ===== some prototype potentials ======
print("Loading some interatomic potentials . .")
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

##
h0("Starting JuLIP Tests")

@testset "JuLIP" begin
   for (testfile, testid) in julip_tests
      h1("Testset $(testid)")
      @testset "$(testid)" begin include(testfile); end
   end
end


# TODO:
# - stillinger-weber => is this not in the hessian tests???
