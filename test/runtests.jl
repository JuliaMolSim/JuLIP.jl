
using JuLIP
using Base.Test
using JuLIP.Testing

verbose=true
isCI = haskey(ENV, "CI")
notCI = !isCI

julip_tests = [
   ("testaux.jl", "Miscellaneous"),
   ("testase.jl", "ASE"),
   ("testanalyticpotential.jl", "Analytic Potential"),
   # ("testpotentials.jl", "Potentials"),
   # ("testvarcell.jl", "Variable Cell"),
   # ("testhessian.jl", "Hessian"),
   # ("testsolve.jl", "Solve"),
]

# remove testsolve if on Travis
if isCI
   julip_tests = julip_tests[1:end-1]
end

# "testexpvarcell.jl";  # USE THIS TO WORK ON EXPCELL IMPLEMENTATION

# ===== some prototype potentials ======
print("Loading some interatomic potentials . .")
data = joinpath(dirname(@__FILE__), "..", "data") * "/"
eam_Fe = JuLIP.Potentials.EAM(data * "pfe.plt", data * "ffe.plt", data * "F_fe.plt")
print(" .")
eam_W = JuLIP.Potentials.FinnisSinclair(data*"W-pair-Wang-2014.plt", data*"W-e-dens-Wang-2014.plt")
print(" .")
if !isCI
   eam_W4 = JuLIP.Potentials.EAM(data * "w_eam4.fs")
end
println(" done.")

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
