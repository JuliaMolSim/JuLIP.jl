using JuLIP
using Base.Test

julip_tests = [
   "testaux.jl";
   "testase.jl"
]

println("Starting JuLIP Tests")
println("=====================")

for test in julip_tests
   include(test)
end
