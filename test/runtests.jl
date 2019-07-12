
using JuLIP, Test, Printf
using JuLIP.Testing

h0("   JuLIP Tests   ")

@info("preparing the tests...")

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
   ("test_fio.jl", "File IO"),
   ("testanalyticpotential.jl", "Analytic Potential"),
   ("testpotentials.jl", "Potentials"),
   ("test_ad.jl", "AD Potentials"),
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

# x = rand()
# sqrt(x)
# eam_W4.ϕ.p2.f(x)
##

@testset "JuLIP" begin
   for (testfile, testid) in julip_tests
      h1("Testset $(testid)")
      @testset "$(testid)" begin include(testfile); end
   end
end
#
#
#
# at = set_pbc!( bulk(:W, cubic = true), false ) * 2
# fdtest(eam_Fe, at, verbose=true)
#
# Testing.fdtest_R2R(r -> eam_W4.ϕ(r), r -> (@D eam_W4.ϕ(r)), 2.5 .+ rand(10))
# Testing.fdtest_R2R(r -> eam_W4.ρ(r), r -> (@D eam_W4.ρ(r)), 2.5 .+ rand(10))
# Testing.fdtest_R2R(r -> eam_W4.F(r), r -> (@D eam_W4.F(r)), 2.5 .+ rand(10))
#
#
# at9 = set_pbc!( bulk(:Fe, cubic = true), false ) * 2
# fdtest(eam_Fe, at9, verbose=true)
