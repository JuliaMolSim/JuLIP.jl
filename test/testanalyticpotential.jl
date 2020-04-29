
import JuLIP.Potentials: @pot, @D, evaluate!, evaluate_d!, PairPotential,
                         @analytic, SimplePairPotential
using LinearAlgebra: norm
using BenchmarkTools
using JuLIP, Test

##

h3("generate hand-coded morse potential")

mutable struct Morseold <: SimplePairPotential
   e0::Float64
   A::Float64
   r0::Float64
end

@pot Morseold

evaluate!(tmp, p::Morseold, r::Number) =
      p.e0 * (exp(-2*p.A*(r/p.r0-1.0)) - 2.0*exp(-p.A*(r/p.r0-1.0)))
evaluate_d!(tmp, p::Morseold, r::Number) =
      -2.0*p.e0*p.A/p.r0*(exp(-2*p.A*(r/p.r0-1.0)) - exp(-p.A*(r/p.r0-1.0)))

A = 4.1
e0 = 0.99
r0 = 1.05

morseold = Morseold(e0, A, r0)

h3("generate AD morse potential")

morse1 = let A = A, e0 = e0, r0 = r0
   @analytic( r -> e0 * (exp(-2*A*(r/r0-1.0)) - 2.0*exp(-A*(r/r0-1.0))) )
end
@show typeof(morse1)


h3("Check consistency of hand-coded and analytic Morse potentials...")
rr = collect(range(0.9, stop=2.1, length=100))
morse_r = morse1.(rr)
morseold_r = morseold.(rr)
dmorse_r = [(@D morse1(r)) for r in rr]
dmorseold_r = [ (@D morseold(r)) for r in rr ]

println(@test norm(morse_r - morseold_r, Inf) < 1e-12)
println(@test norm(dmorse_r - dmorseold_r, Inf) < 1e-12)

##

function runn(f, x, N)
   s = 0.0
   for n = 1:N
      x += 0.0001
      s += f(x)
   end
   return s
end

function runn_d(f, x, N)
   s = 0.0
   for n = 1:N
      x += 0.0001
      s += @D f(x)
   end
   return s
end


h3("Performance tests: @analytic vs hand-coded")
x = 1.0+rand()
println("Evaluations of @analytic Potential")
@btime runn($morse1, $x, 1_000)
println("Evaluations hand-coded Potential")
@btime runn($morseold, $x, 1_000)
println("Grad of @analytic Potential")
@btime runn_d($morse1, $x, 1_000)
println("Grad of hand-coded Potential")
@btime runn_d($morseold, $x, 1_000)
