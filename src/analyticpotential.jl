
# wait for ReverseDiffSource to be updated to v0.5
# import ReverseDiffSource
# import ReverseDiffSource:rdiff

import Calculus
import Calculus: differentiate
import FunctionWrappers
import FunctionWrappers: FunctionWrapper

typealias F64fun FunctionWrapper{Float64, Tuple{Float64}}


export AnalyticPotential

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
#       but need a better model


# documentation attached below
@pot type AnalyticPotential{F,F1,F2} <: PairPotential
   f::F
   f_d::F1
   f_dd::F2
   id::AbstractString
   cutoff::Float64
end

"""
`type AnalyticPotential <: PairPotential`

### Usage:
```julia
lj = AnalyticPotential(:(r^(-12) - 2.0*r^(-6)), "LennardJones")
println(lj)   # will output `LennardJones`
A = 4.0; r0 = 1.234
morse = AnalyticPotential("exp(-2.0*\$A*(r/\$r0-1.0)) - 2.0*exp(-\$A*(r/\$r0-1.0))",
                           "Morse(A=\$A,r0=\$r0)")
```

use kwarg `cutoff` to set a cut-off, default is `Inf`
"""
AnalyticPotential

evaluate(p::AnalyticPotential, r) = p.f(r)
evaluate_d(p::AnalyticPotential, r) = p.f_d(r)
evaluate_dd(p::AnalyticPotential, r) = p.f_dd(r)
Base.print(io::Base.IO, p::AnalyticPotential) = print(io, p.id)
cutoff(p::AnalyticPotential) = p.cutoff

# construct from string or expression
AnalyticPotential(s::AbstractString; id = s, cutoff=Inf) =
                        AnalyticPotential(parse(s), id=id, cutoff=cutoff)

function AnalyticPotential(ex::Expr; id = string(ex), cutoff=Inf)
   @assert isa(id, AbstractString)
   f, f_d, f_dd = diff2(ex)
   return AnalyticPotential(f, f_d, f_dd, id, cutoff)
end

# this is a hack to make tight-binding work; but it should be reconsidered
# right now I am thinking it is actually ok as is!
evaluate(p::AnalyticPotential, r, R) = evaluate(p, r)
evaluate_d(p::AnalyticPotential, r, R) = evaluate_d(p, r)
