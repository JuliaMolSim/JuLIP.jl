
using JuLIP
using Base.Test
using JuLIP.Testing


verbose=true

julip_tests = [
<<<<<<< HEAD
  #  "testaux.jl";
  #  "testase.jl";
  #  "testanalyticpotential.jl";
  #  "testpotentials.jl";
  # "testvarcell.jl";
  # "testsolve.jl";
  "testhessian.jl"
=======
   "testaux.jl";
   "testase.jl";
   "testdft.jl";
   "testanalyticpotential.jl";
   "testpotentials.jl";
   "testvarcell.jl";
   "testexpvarcell.jl";
   "testsolve.jl";
>>>>>>> bc2a2a1a8df031438407bf3449bc8ba7ad248587
]

println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("  Starting JuLIP Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")


for test in julip_tests
   include(test)
end
