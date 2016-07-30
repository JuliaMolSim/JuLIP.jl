"""
## module Potentials

### Summary

This module implements some basic interatomic potentials in pure Julia, as well
as provides building blocks and prototypes for further implementations
The implementation is done in such a way that they can be used either in "raw"
form or within abstract frameworks.

### Types

### `evaluate`, `evaluate_d`, `evaluate_dd`, `grad`

### The `@D`, `@DD`, `@GRAD` macros

TODO: write documentation

"""
module Potentials

import JuLIP: AbstractAtoms, AbstractNeighbourList, AbstractCalculator,
   bonds, sites,
   energy, forces, cutoff,
   JVec, JVecs, JPt, JPts, mat, pts, vec

# we also import grad from JuLIP, but to define derivatives
import JuLIP: grad


export LennardJonesPotential, MorsePotential, SimpleExponential, ZeroPairPotential,
   SWCutoff, ShiftCutoff, SplineCutoff,
   LennardJonesCalculator, MorseCalculator, GuptaCalculator,
   @D, @DD, @GRAD

"""
`Potential`: generic abstract supertype for all potential-like things
"""
abstract Potential

"""
`PairPotential`: abstract supertype for pair potentials
"""
abstract PairPotential <: Potential

"""
An `AbstractCutoff` is a type that, when evaluated will return the cut-off
variant of that potential.

All `Cutoff <: AbstractCutoff` must have a constructor of the form
```
Cutoff(pp::PairPotential, args...)
```
"""
abstract AbstractCutoff <: PairPotential

"""
`abstract SitePotential <: Potential`

TODO: write documentation
"""
abstract SitePotential <: Potential


# ================== Analytical Potentials ==========================

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

use kwarg `cutoff` to set a cut-off, default is `Inf`

TODO: this should not be restricted to pair potentials
"""
type AnalyticPotential{T} <: PairPotential
   v::T
   id::AbstractString
   cutoff::Float64
end

# display the potential
Base.print(io::Base.IO, p::AnalyticPotential) = print(io, p.id)
Base.show(io::Base.IO, p::AnalyticPotential) = print(io, p.id)
cutoff(p::AnalyticPotential) = p.cutoff

# construct from string or expression
AnalyticPotential(s::AbstractString; id = s, cutoff=Inf) = AnalyticPotential(parse(s), id=id, cutoff=Inf)

function AnalyticPotential(ex::Expr; id = string(ex), cutoff=Inf)
   @assert typeof(id) <: AbstractString
   # differentiate the expression
   dex = ReverseDiffSource.rdiff(ex, r=1.0, allorders=false)
   # overload the two evaluation functions
   eval( quote
      evaluate(ap::AnalyticPotential{Val{Symbol($id)}}, r::Float64) = $ex
      evaluate_d(ap::AnalyticPotential{Val{Symbol($id)}}, r::Float64) = $dex
   end )
   return AnalyticPotential(Val{Symbol(id)}(), id, cutoff)
end



include("potentialarithmetic.jl")




# ===========================================================================
#     implement some fun little macros for easier access
#     to the potentials
# ===========================================================================

# First we create aliases
#   * call(p, ...) = evaluate(p, ...)
#     this allows e.g. p(r) in lieu of evaluate(p, r)
#   * call(p, Val{:D}, ...) = evaluate_d(p, ...)
#   * call(p, Val{:DD}, ...) = evaluate_dd(p, ...)

import Base.call
@inline call(pp::Potential, varargs...) = evaluate(pp, varargs...)
@inline call(pp::Potential, ::Type{Val{:D}}, varargs...) = evaluate_d(pp, varargs...)
@inline call(pp::Potential, ::Type{Val{:DD}}, varargs...) = evaluate_dd(pp, varargs...)
@inline call(pp::Potential, ::Type{Val{:GRAD}}, varargs...) = grad(pp, varargs...)

# The call aliases make the following macros possible: they basically just
# sneak an extra parameter into the `call`

# next create macros that translate
"""
`@D`: Use to evaluate the derivative of a potential. E.g., to compute the
Lennard-Jones potential,
```julia
lj = LennardJonesPotential()
r = 1.0 + rand(10)
ϕ = lj(r)
ϕ' = @D lj(r)
```
see also `@DD`.
"""
macro D(fsig::Expr)
    @assert fsig.head == :call
    insert!(fsig.args, 2, Val{:D})
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    return fsig
end

"`@DD` : analogous to `@D`"
macro DD(fsig::Expr)
    @assert fsig.head == :call
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    insert!(fsig.args, 2, Val{:DD})
    return fsig
end

"`@GRAD` : analogous to `@D`, but escapes to `grad`"
macro GRAD(fsig::Expr)
    @assert fsig.head == :call
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    insert!(fsig.args, 2, Val{:GRAD})
    return fsig
end


# ===========================================================================
#    cutoff potentials
# ===========================================================================

include("cutoffs.jl")
#   * SWCutoff
#   * ShiftCutoff
#   * SplineCutoff


# ===========================================================================
#    Pair potentials
# ===========================================================================

include("pairpotentials.jl")
# * PairCalculator
# * LennardJonesPotential
# * MorsePotential
# * SimpleExponential


################### the EMT Calculator ###################

# include("emt.jl")



end
