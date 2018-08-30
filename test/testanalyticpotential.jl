
import JuLIP.Potentials: @pot, @D, evaluate, evaluate_d, PairPotential, @analytic

@pot mutable struct Morseold <: PairPotential
   e0::Float64
   A::Float64
   r0::Float64
end
evaluate(p::Morseold, r) = p.e0 * (exp(-2*p.A*(r/p.r0-1.0)) - 2.0*exp(-p.A*(r/p.r0-1.0)))
evaluate_d(p::Morseold, r) = -2.0*p.e0*p.A/p.r0*(exp(-2*p.A*(r/p.r0-1.0)) - exp(-p.A*(r/p.r0-1.0)))

const e0 = 0.99
const r0 = 1.05
const A = 4.1

morse = @analytic( r -> e0 * (exp(-2*A*(r/r0-1.0)) - 2.0*exp(-A*(r/r0-1.0))) )
@show typeof(morse)
morseold = Morseold(e0, A, r0)

rr = collect(range(0.9, stop=2.1, length=100))
morse_r = morse.(rr)
morseold_r = morseold.(rr)
dmorse_r = [(@D morse(r)) for r in rr]
dmorseold_r = [ (@D morseold(r)) for r in rr ]

println("Check consistency of hand-coded and analytic Morse potentials...")
@test vecnorm(morse_r - morseold_r, Inf) < 1e-12
@test vecnorm(dmorse_r - dmorseold_r, Inf) < 1e-12

function test(N, p)
   r = 1.234
   s = 0.0
   for n = 1:N
      s += p(r)
   end
   return s
end

function testgrad(N, p)
   r = 1.234
   s = 0.0
   for n = 1:N
      s += @D p(r)
   end
   return s
end

test(10, morse);
test(10, morseold);
testgrad(10, morse);
testgrad(10, morseold);

println("1M evaluations of Analytic Potential")
@time test(1_000_000, morse)
println("1M evaluations of old Potential")
@time test(1_000_000, morseold)
println("1M ∇ evaluations of Analytic Potential")
@time testgrad(1_000_000, morse)
println("1M ∇ evaluations of old Potential")
@time testgrad(1_000_000, morseold)
