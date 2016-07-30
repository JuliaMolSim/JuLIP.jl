
import ReverseDiffSource

"""
`type AnalyticPotential <: PairPotential`

### Usage:
```julia
lj = AnalyticPotential(:(r^(-12) - 2.0*r^(-6)), "LennardJones")
A = 4.0; r0 = 1.234
morse = AnalyticPotential("exp(-2.0*\$A*(r/\$r0-1.0)) - 2.0*exp(-\$A*(r/\$r0-1.0))",
                           "Morse(A=\$A,r0=\$r0)")
```

TODO: this should not be restricted to pair potentials
"""
type AnalyticPotential{T} <: PairPotential
   v::T
   id::AbstractString
end

# display the potential
Base.print(io::Base.IO, p::AnalyticPotential) = print(io, p.id)
Base.show(io::Base.IO, p::AnalyticPotential) = print(io, p.id)

# construct from string or expression
AnalyticPotential(s::AbstractString; id = s) = AnalyticPotential(parse(s), id=id)

function AnalyticPotential(ex::Expr; id = string(ex))
   @assert typeof(id) <: AbstractString
   # differentiate the expression
   dex = ReverseDiffSource.rdiff(ex, r=1.0, allorders=false)
   # overload the two evaluation functions
   eval( quote
      evaluate(ap::AnalyticPotential{Val{Symbol($id)}}, r::Float64) = $ex
      evaluate_d(ap::AnalyticPotential{Val{Symbol($id)}}, r::Float64) = $dex
   end )
   return AnalyticPotential(Val{Symbol(id)}(), id)
end


# # scalars as potentials
# evaluate(x::Real, r::Float64) = x
# evaluate_d(x::Real, r::Float64) = 0.0
#
#
# "basic building block to generate potentials"
# type r_Pot <: PairPotential end
# evaluate(p::r_Pot, r) = r
# evaluate_d(p::r_Pot, r) = 1
#
#
# "sum of two pair potentials"
# type sum_Pot{P1, P2}
#    p1::P1
#    p2::P2
# end
#
# typealias PPorNum Union{PairPotential, Real}
#
# (Base.+)(p1::PairPotential, p2::PPorNum) = sum_Pot(p1, p2)
# (Base.+)(p2::PPorNum, p1::PairPotential) = sum_Pot(p1, p2)
#
# evaluate(p::sum_Pot, r) = p.p1(r) + p.p2(r)
# evaluate_d(p::sum_Pot, r) = (@D p.p1(r)) + (@D p.p2(r))
#
# "product of two pair potentials"
# type prod_Pot{P1, P2}
#    p1::P1
#    p2::P2
# end
#
# (Base.*)(p1::PairPotential, p2::PPorNum) = prod(p1, p2)
# (Base.*)(p1::PPorNum, p2::PairPotential) = prod(p1, p2)
#
# evaluate(p::prod_Pot, r) = p.p1(r) * p.p2(r)
# evaluate_d(p::prod_Pot, r) = p.p1(r) * (@D p.p2(r)) + (@D p.p1(r)) * p.p2(r)
#
#
# # "product of two pair potentials"
# # type exp_Pot{P1}
# #    p1::P1
# # end
# #
# # Base.exp(p1::PairPotential) = exp_Pot(p1)
# #
# # evaluate(p::exp_Pot, r) = exp(p.p1(r)))
# # evaluate_d(p::exp_Pot, r) = exp(p.p1(r)) * (@D p.p1(r))
#
#
# const r = r_Pot()
