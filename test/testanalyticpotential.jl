
import JuLIP.Potentials: @D, evaluate, evaluate_d, PairPotential

type LJold <: PairPotential
    e0::Float64
   r0::Float64
end
evaluate(p::LJold, r) = p.e0 * ( (r/p.r0).^(-12) - 2.0 * (r/p.r0).^(-6) )
evaluate_d(p::LJold, r) = -12.0*p.e0/p.r0 * ( (r/p.r0).^(-13) - (r/p.r0).^(-7) )

r0 = 1.1
e0 = 0.9

lj = JuLIP.Potentials.AnalyticPotential(
                     "$e0 * ((r/$r0)^(-12) - 2.0 * (r/$r0)^(-6))",
                     id="lj(e0=$e0,r0=$r0)")
# lj = JuLIP.Potentials.LennardJonesPotential(e0=e0, r0=r0)
@show typeof(lj)
ljold = LJold(e0, r0)

rr = collect(linspace(0.9, 2.1, 100))
lj_r = [lj(r) for r in rr]
ljold_r = ljold(rr)
dlj_r = [(@D lj(r)) for r in rr]
dljold_r = @D ljold(rr)

@show vecnorm(lj_r - ljold_r, Inf)
@show vecnorm(dlj_r - dljold_r, Inf)

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
@time test(1_000_000, lj)
println("1M ∇ evaluations of old Potential")
@time test(1_000_000, ljold)
