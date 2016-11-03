
# wait for ReverseDiffSource to be updated to v0.5
# import ReverseDiffSource
# import ReverseDiffSource:rdiff

using Calculus: differentiate

export AnalyticPairPotential, AnalyticPotential, WrappedPPotential

import FunctionWrappers
import FunctionWrappers: FunctionWrapper
typealias F64fun FunctionWrapper{Float64, Tuple{Float64}}


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

"""
`diff2_wrapF64(ex::Expr)` :

takes an expression, differentiates it twice,
     and returns anonymous functions for the derivatives,
     wrapped using `FunctionWrappers`.
"""
function diff2_wrapF64(ex::Expr, sym=:r)
   f, f_d, f_dd = diff2(ex, sym)
   return F64fun(f), F64fun(f_d), F64fun(f_dd)
end



# ================== Analytical Potentials ==========================
#
# TODO: this construction should not be restricted to pair potentials
#       but need a better model

abstract AnalyticPairPotential <: PairPotential

# documentation attached below
@pot type AnalyticPotential{F0,F1,F2} <: AnalyticPairPotential
   f::F0
   f_d::F1
   f_dd::F2
   id::AbstractString
   cutoff::Float64
end

"""
`type AnalyticPotential <: AnalyticPairPotential`

### Usage:
```julia
lj = PairPotential(:(r^(-12) - 2.0*r^(-6)), "LennardJones")
println(lj)   # will output `LennardJones`
A = 4.0; r0 = 1.234
morse = PairPotential("exp(-2.0*\$A*(r/\$r0-1.0)) - 2.0*exp(-\$A*(r/\$r0-1.0))",
                           "Morse(A=\$A,r0=\$r0)")
```

use kwarg `cutoff` to set a cut-off, default is `Inf`
"""
AnalyticPotential

# documentation attached below
@pot type WrappedPPotential <: AnalyticPairPotential
   f::F64fun
   f_d::F64fun
   f_dd::F64fun
   id::AbstractString
   cutoff::Float64
end

"""
`type WrappedPPotential <: AnalyticPairPotential`

similar to `AnalyticPotential`, but using `FunctionWrappers` so that
these potentials can be stored in an array without performance penalty.
"""
WrappedPPotential



evaluate(p::AnalyticPairPotential, r) = p.f(r)
evaluate_d(p::AnalyticPairPotential, r) = p.f_d(r)
evaluate_dd(p::AnalyticPairPotential, r) = p.f_dd(r)
Base.print(io::Base.IO, p::AnalyticPairPotential) = print(io, p.id)
cutoff(p::AnalyticPairPotential) = p.cutoff

# construct from string or expression
AnalyticPotential(s::AbstractString; id = s, cutoff=Inf) =
      AnalyticPotential(parse(s), id=id, cutoff=cutoff)

function AnalyticPotential(ex::Expr; id::AbstractString = string(ex), cutoff=Inf)
   warn("""`AnalyticPotential` is deprecated;
            please use the `PairPotential` constructor instead""")
   return PairPotential(ex, id=id, cutoff=cutoff)
end


PairPotential(s::AbstractString; id = s, cutoff=Inf) =
      PairPotential(parse(s), id=id, cutoff=cutoff)

PairPotential(ex::Expr; id::AbstractString = string(ex), cutoff=Inf) =
   AnalyticPotential(diff2(ex)..., id, cutoff)


WrappedPPotential(s::AbstractString; id = s, cutoff=Inf) =
      WrappedPPotential(parse(s), id=id, cutoff=cutoff)

WrappedPPotential(ex::Expr; id::AbstractString = string(ex), cutoff=Inf) =
   WrappedPPotential(diff2_wrapF64(ex)..., id, cutoff)



# this is a hack to make tight-binding work; but it should be reconsidered
# right now I am thinking it is actually ok as is!
evaluate(p::AnalyticPairPotential, r, R) = evaluate(p, r)
evaluate_d(p::AnalyticPairPotential, r, R) = evaluate_d(p, r)
