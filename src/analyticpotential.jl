
using MacroTools
using Calculus: differentiate

export AnalyticPairPotential, @PairPotential

import FunctionWrappers
import FunctionWrappers: FunctionWrapper
const ScalarFun{T} = FunctionWrapper{T, Tuple{T}}

# ===========================================================================
#     macro that takes an expression, differentiates it
#     and returns anonymous functions for the derivatives
# ===========================================================================


"""
`diff2(ex::Expr)` :

takes an expression, differentiates it twice,
     and returns anonymous functions for the derivatives
"""
function diff2(ex::Expr, sym=:r)
   ex_d = differentiate(ex, sym)
   ex_dd = differentiate(ex_d, sym)
   return eval( :( (r->$ex, r->$ex_d, r->$ex_dd)) )
end


# ================== Analytical Potentials ==========================
#
# TODO: this construction should not be restricted to pair potentials
#       but need a better model for how to implement this generically

# documentation attached below
@pot struct AnalyticPairPotential{F0,F1,F2} <: PairPotential
   f::F0
   f_d::F1
   f_dd::F2
   cutoff::Float64
end


"""
`type AnalyticPairPotential <: PairPotential`

### Usage:
```julia
lj = @PairPotential r -> r^(-12) - 2.0*r^(-6)
A = 4.0; r0 = 1.234
morse = let A = 4.0, r0 = 1.234
   @PairPotential r -> exp(-2.0*A*(r/r0-1.0)) - 2.0*exp(-A*(r/r0-1.0))
end

To create a "wrapped" AnalyticPairPotential{F64fun, ...} , use
```
lj_wrapped = F64fun(lj)
```
"""
AnalyticPairPotential


"""
`F64fun`: `FunctionWrapper` to wrap many different potentials within a single
type. Can be used as  `F64fun(::Function)` or as
`F64fun(::AnalyticPairPotential)`
"""
const F64fun = ScalarFun{Float64}

F64fun(p::AnalyticPairPotential) =
   AnalyticPairPotential(F64fun(p.f), F64fun(p.f_d), F64fun(p.f_dd), cutoff)


evaluate(p::AnalyticPairPotential, r::Number) = p.f(r)
evaluate_d(p::AnalyticPairPotential, r::Number) = p.f_d(r)
evaluate_dd(p::AnalyticPairPotential, r::Number) = p.f_dd(r)
cutoff(p::AnalyticPairPotential) = p.cutoff

function fdiff( ex )
   @assert @capture(ex, var_ -> expr_)
   return :(  $var -> $(differentiate(expr.args[2], var)) )
end

"""
`@PairPotential`: generate symbolic pair potentials.
"""
macro PairPotential(fexpr)
   :(
      AnalyticPairPotential(
        $(esc(fexpr)),
        $(esc(fdiff(fexpr))),
        $(esc(fdiff(fdiff(fexpr)))),
        Inf
      )
   )
end


# this is a hack to make tight-binding work; but it should be reconsidered
# right now I am thinking it is actually ok as is!
evaluate(p::AnalyticPairPotential, r, R) = evaluate(p, r)
evaluate_d(p::AnalyticPairPotential, r, R) = evaluate_d(p, r)
