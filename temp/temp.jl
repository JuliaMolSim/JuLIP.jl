
# benchmarking different ways to implement AnalyticPotential,
# in order to circumvent the World Age problem

using MacroTools
using BenchmarkTools
using Calculus: differentiate
using FunctionWrappers: FunctionWrapper
const F64fun = FunctionWrapper{Float64, Tuple{Float64}}

function Base.diff(ex::Expr)
   ex_d = differentiate(ex, :r)
   return eval( :( (r->$ex, r->$ex_d)) )
end

struct AnalyticPotential{F0,F1}
   f::F0
   f_d::F1
end

macro AnalyticPotential(expr)
   :(
      AnalyticPotential(
        $(esc(:(r->$(expr)))),
        $(esc(:(r->$(differentiate(expr,:r)))))
      )
   )
end

macro AnalyticPotential(fexpr)
   :(
      AnalyticPotential(
        $(esc(fexpr)),
        $(esc(fdiff(fexpr)))
      )
   )
end

function fdiff( ex )
   @assert @capture(ex, var_ -> expr_)
   return :( $var -> $(differentiate(expr.args[2], var)) )
end

function reftest(N)
   s0 = 0.0
   s1 = 0.0
   for n = 1:N
      x = rand()
      s0 += exp( - 1.3711 * x + 1.234 )
      s1 += (- 1.3711) * exp( - 1.3711 * x + 1.234 )
   end
end

function innertest(p::AnalyticPotential, N)
   s0 = 0.0
   s1 = 0.0
   for n = 1:N
      x = rand()
      s0 += p.f(x)
      s1 += p.f_d(x)
   end
end


Nruns = 1_000_000
const a = 1.234
const r0 = 0.9

# REFERENCE
println("Reference Timing")
reftest(10)
@btime reftest($Nruns)

# OLD VERSION (as script so no world problem)
println("AnalyticPotential( diff(Expr) ) from REPL")
p = AnalyticPotential( diff(:( exp(- $a * (r*$(1/r0) - 1.0)) ))... )
innertest(p, 10)
@btime innertest($p, $Nruns)

# OLD VERSION, slightly optimised to match the reference
println("AnalyticPotential, slightly optimised")
po = AnalyticPotential( diff(:( exp(- $(a/r0) * r + $a) ) )... )
innertest(po, 10)
@btime innertest($po, $Nruns)

# Function Wrappers
println("Function Wrappers")
pfw = AnalyticPotential( F64fun(p.f), F64fun(p.f_d) )
innertest(pfw, 10)
@btime innertest($pfw, $Nruns)

# invokelatest
println("Invokelatest variant")
function testil(N)
   p = AnalyticPotential( diff(:( exp(- $(a/r0) * r + $a) ) )... )
   Base.invokelatest(innertest, p, 10)
   @btime Base.invokelatest(innertest, $p, $N)
end
testil(Nruns)


println("lambdas")
pl = AnalyticPotential( r -> exp(- (a/r0) * r + a),
               r -> - (a/r0) * exp(- (a/r0) * r + a) )
innertest(pl, 10)
@btime innertest($pl, $Nruns)



println("Macro variant")
function testm(N, a, r0)
   p = @AnalyticPotential r -> exp(- (a/r0) * r + a)
   innertest(p, 10)
   @btime innertest($p, $N)
end
testm(Nruns, a, r0)
