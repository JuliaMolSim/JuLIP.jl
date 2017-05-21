

verbose=true


julip_tests = [
   "testaux.jl";
   "testase.jl";
   "testdft.jl";
   "testanalyticpotential.jl";
   "testpotentials.jl";
   "testvarcell.jl";
   "testhessian.jl";
   "testsolve.jl";
]
# "testexpvarcell.jl";  # USE THIS TO WORK ON EXPCELL IMPLEMENTATION

println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("  Starting JuLIP Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")


@everywhere using JuLIP
@everywhere using Base.Test
@everywhere using JuLIP.Testing


if haskey(ENV, "CI")
   @show ENV["CI"]
end


# for test in julip_tests
#    include(test)
# end
