
using JuLIP
using Base.Test
using JuLIP.Testing

verbose=true

julip_tests = [
   # ("testaux.jl", "Miscellaneous"),
   # ("testase.jl", "ASE"),
   # ("testdft.jl", "DFT"),
   # ("testanalyticpotential.jl", "Analytic Potential"),
   # ("testpotentials.jl", "Potentials"),
   # ("testvarcell.jl", "Variable Cell"),
   # ("testhessian.jl", "Hessian"),
   ("testsolve.jl", "Solve"),
]
# "testexpvarcell.jl";  # USE THIS TO WORK ON EXPCELL IMPLEMENTATION


# ===== some prototype potentials ======
data = joinpath(dirname(@__FILE__), "..", "data") * "/"
eam_Fe = JuLIP.Potentials.EAM(data * "pfe.plt", data * "ffe.plt", data * "F_fe.plt")
eam_W = JuLIP.Potentials.FinnisSinclair(data*"W-pair-Wang-2014.plt", data*"W-e-dens-Wang-2014.plt")


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


# TODO: if we want to check whether we are on travis then we can use this:
# if haskey(ENV, "CI")
#    @show ENV["CI"]
# end
# quit()
