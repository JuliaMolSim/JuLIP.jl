
using JuLIP
using Base.Test
using JuLIP.Testing


verbose=true

julip_tests = [
   # "testaux.jl";
   # "testase.jl";
   # "testdft.jl";
   # "testanalyticpotential.jl";
   # "testpotentials.jl";
   # "testvarcell.jl";
   # "testexpvarcell.jl";
   "testsolve.jl";
   "testhessian.jl"
]

println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("  Starting JuLIP Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")


for test in julip_tests
   include(test)
end
