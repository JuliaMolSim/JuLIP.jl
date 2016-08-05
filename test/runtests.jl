
using JuLIP
using Base.Test
using JuLIP.Testing

verbose=true

julip_tests = [
   "testaux.jl";
   "testase.jl";
   # "testanalyticpotential.jl"
   "testpotentials.jl";
   "testsolve.jl";
]

println("Starting JuLIP Tests")
println("=====================")

for test in julip_tests
   include(test)
end
