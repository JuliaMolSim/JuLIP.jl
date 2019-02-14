
using MacroTools: @capture, prewalk
using Calculus: differentiate

export AnalyticFunction, @analytic
using CommonSubexpressions

import FunctionWrappers
import FunctionWrappers: FunctionWrapper
const ScalarFun{T} = FunctionWrapper{T, Tuple{T}}

"""
`F64fun`: `FunctionWrapper` to wrap many different potentials within a single
type. Can be used as  `F64fun(::Function)` or as
`F64fun(::AnalyticFunction)`
"""
const F64fun = ScalarFun{Float64}


"""
take an expression of the form `r -> f(r)` and return the expression
`r -> f'(r)`
"""
function fdiff( ex, ndiff )
   @assert 1 <= ndiff <= 2
   @assert @capture(ex, var_ -> expr_)
   d_ex = differentiate(expr.args[2], var)
   if ndiff == 2
      d_ex = differentiate(d_ex, var)
   end
   d_ex = CommonSubexpressions.cse(d_ex)
   return :(  $var -> $d_ex )
end

"""
auxiliary function to allow expression substitution from `@Analytic`;
see `?@Analytic` for more detail
"""
function substitute(args)
    subs = Dict{Symbol,Union{Expr, Symbol}}()
    for i = 2:length(args)
        @assert @capture(args[i], var_ = sub_)
        subs[var] = sub
    end
    prewalk(expr -> get(subs, expr, expr), args[1])
end

# ================== Analytical Potentials ==========================

"""
`struct AnalyticFunction`: described an analytic function, allowing to
evaluate at least 2 derivatives.

Formally, `AnalyticFunction <: PairPotential`, which simplifies dispatch,
but it an `AnalyticFunction` should normally **not be used as a Calculator**!
This type hierarchy may need to be revisited.

### Usage:
```julia
lj = @analytic r -> r^(-12) - 2.0*r^(-6)
morse = let A = 4.0, r0 = 1.234
   @analytic r -> exp(-2.0*A*(r/r0-1.0)) - 2.0*exp(-A*(r/r0-1.0))
end

To create a "wrapped" AnalyticFunction{F64fun, ...} , use
```
lj_wrapped = F64fun(lj)
```

if an formula for an analytic function contains a sub-expression that
occurs multiple time, then this can be constructed as follows:
```
V = @analytic( r -> exp(s) * s, s = r^2 )
# is the same as
V = @analytic r -> exp(r^2) * r^2
```
"""
struct AnalyticFunction{F0,F1,F2} <: PairPotential
   f::F0
   f_d::F1
   f_dd::F2
end

@pot AnalyticFunction

const WrappedAnalyticFunction = AnalyticFunction{F64fun, F64fun, F64fun}

F64fun(p::AnalyticFunction) =
   AnalyticFunction(F64fun(p.f), F64fun(p.f_d), F64fun(p.f_dd))

evaluate(p::AnalyticFunction, r::Number) = p.f(r)
evaluate_d(p::AnalyticFunction, r::Number) = p.f_d(r)
evaluate_dd(p::AnalyticFunction, r::Number) = p.f_dd(r)
cutoff(V::AnalyticFunction) = Inf


"""
`@analytic`: generate C2 function from symbol
"""
macro analytic(args...)
   fexpr = substitute(args)
   quote
      AnalyticFunction(
        $(Base.FastMath.make_fastmath(esc(fexpr))),
        $(Base.FastMath.make_fastmath(esc(fdiff(fexpr, 1)))),
        $(Base.FastMath.make_fastmath(esc(fdiff(fexpr, 2))))
      )
   end
end


# TODO: this is a hack to make tight-binding work; but it should be reconsidered
# right now I am thinking it is actually ok as is!
evaluate(p::AnalyticFunction, r, R) = evaluate(p, r)
evaluate_d(p::AnalyticFunction, r, R) = evaluate_d(p, r)
