
verbose=true

julip_tests = [
   ("testaux.jl", "Miscellaneous"),
   ("testase.jl", "ASE"),
   # ("testdft.jl", "DFT"),
   # ("testanalyticpotential.jl", "Analytic Potential"),
   # ("testpotentials.jl", "Potentials"),
   # ("testvarcell.jl", "Variable Cell"),
   ("testhessian.jl", "Hessian"),
   # ("testsolve.jl", "Solve"),
]
# "testexpvarcell.jl";  # USE THIS TO WORK ON EXPCELL IMPLEMENTATION

println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("  Starting JuLIP Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")

using JuLIP
using Base.Test
using JuLIP.Testing

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
