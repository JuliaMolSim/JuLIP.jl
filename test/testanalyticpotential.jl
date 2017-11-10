
import JuLIP.Potentials: @pot, @D, evaluate, evaluate_d, PairPotential

@pot type Morseold <: PairPotential
   e0::Float64
   A::Float64
   r0::Float64
end
evaluate(p::Morseold, r) = p.e0 * (exp(-2*p.A*(r/p.r0-1.0)) - 2.0*exp(-p.A*(r/p.r0-1.0)))
evaluate_d(p::Morseold, r) = -2.0*p.e0*p.A/p.r0*(exp(-2*p.A*(r/p.r0-1.0)) - exp(-p.A*(r/p.r0-1.0)))

const e0 = 0.99
const r0 = 1.05
const A = 4.1

lj = JuLIP.Potentials.PairPotential(
                     "$e0 * (exp(-2*$A*(r/$r0-1.0)) - 2.0*exp(-$A*(r/$r0-1.0)))",
                     id="Morse(e0=$e0,r0=$r0,A=$A)")
@show typeof(lj)
ljold = Morseold(e0, A, r0)

rr = collect(linspace(0.9, 2.1, 100))
lj_r = lj.(rr)
ljold_r = ljold.(rr)
dlj_r = [(@D lj(r)) for r in rr]
dljold_r = [ (@D ljold(r)) for r in rr ] 

@test vecnorm(lj_r - ljold_r, Inf) < 1e-14
@test vecnorm(dlj_r - dljold_r, Inf) < 1e-14

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

test(10, lj);
test(10, ljold);
testgrad(10, lj);
testgrad(10, ljold);

println("1M evaluations of Analytic Potential")
@time test(1_000_000, lj)
println("1M evaluations of old Potential")
@time test(1_000_000, ljold)
println("1M ∇ evaluations of Analytic Potential")
@time testgrad(1_000_000, lj)
println("1M ∇ evaluations of old Potential")
@time testgrad(1_000_000, ljold)
