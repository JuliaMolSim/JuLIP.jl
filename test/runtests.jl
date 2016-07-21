
using JuLIP
using Base.Test
using JuLIP.Testing

verbose=true

julip_tests = [
   # "testaux.jl";
   # "testase.jl"
   "testpotentials.jl"
]

println("Starting JuLIP Tests")
println("=====================")

for test in julip_tests
   include(test)
end
