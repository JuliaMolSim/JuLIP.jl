
using JuLIP
using Base.Test
using JuLIP.Testing

verbose=true

## check whether on CI
isCI = haskey(ENV, "CI")
notCI = !isCI
eam_W4 = nothing

## check whether ASE is available
hasase = true
try
   import ASE
catch
   hasase = false
end

julip_tests = [
   ("testaux.jl", "Miscellaneous"),
   ("test_atoms.jl", "Atoms"),
   ("test_build.jl", "Build"),
   ("testanalyticpotential.jl", "Analytic Potential"),
   ("testpotentials.jl", "Potentials"),
   # ("test_ad.jl", "AD Potentials"),
   ("testvarcell.jl", "Variable Cell"),
   ("testhessian.jl", "Hessian"),
   ("testsolve.jl", "Solve"),
]

# remove testsolve if on Travis
if isCI
   julip_tests = julip_tests[1:end-1]
end

# "testexpvarcell.jl";  # USE THIS TO WORK ON EXPCELL IMPLEMENTATION

## ===== some prototype potentials ======
print("Loading some interatomic potentials . .")
data = joinpath(Pkg.dir("JuLIP"), "data") * "/"
eam_Fe = JuLIP.Potentials.EAM(data * "pfe.plt", data * "ffe.plt", data * "F_fe.plt")
print(" .")
eam_W = JuLIP.Potentials.FinnisSinclair(data*"W-pair-Wang-2014.plt", data*"W-e-dens-Wang-2014.plt")
print(" .")
try
   eam_W4 = JuLIP.Potentials.EAM(data * "w_eam4.fs")
catch
   eam_W4 = nothing
end
println(" done.")

##
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("  Starting JuLIP Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")

@testset "JuLIP" begin
   for (testfile, testid) in julip_tests
      println("=======================")
      println("Testset $(testid)")
      println("=======================")
      @testset "$(testid)" begin include(testfile); end
   end
end
