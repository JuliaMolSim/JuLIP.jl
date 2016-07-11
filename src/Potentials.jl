"""
## module Potentials

### Summary

This module implements some basic interatomic potentials in pure Julia, as well
as provides building blocks and prototypes for further implementations
The implementation is done in such a way that they can be used either in "raw"
form or within abstract frameworks.

### Types



### The `@D` and `@DD` macros

TODO: write documentation

"""
module Potentials

import JuLIP: AbstractAtoms, AbstractNeighbourList, AbstractCalculator,
   bonds, sites,
   energy, forces, cutoff

# we also import grad from JuLIP, but to define derivatives
import JuLIP: grad

# implement:
#    evaluate
#    evaluate_d
#    evaluate_dd
#    grad

export LennardJonesCalculator, MorseCalculator, GuptaCalculator

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

abstract SitePotential <: Potential


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



# ===========================================================================
#    Site potentials
#    TODO: everything below needs to be tested still!
# ===========================================================================

# "`ZeroSitePotential`: Site potential V(R) = 0.0"
# type ZeroSitePotential <: SitePotential end
# evaluate(p::ZeroSitePotential, R) = 0.0
# evaluate_d(p::ZeroSitePotential, R) = zeros(size(R))
# evaluate(p::ZeroSitePotential, r, R) = 0.0
# evaluate_d(p::ZeroSitePotential, r, R) = zeros(size(R,1),1,1,size(R,2))
# grad(p::ZeroSitePotential, R) = zeros(3,1,1,size(R,1))
# grad(p::ZeroSitePotential, r, R) = zeros(3,1,1,size(R,2))
# cutoff(p::ZeroSitePotential) = 0.0
#
# """
# `EAMPotential`: implementation of the EAM potential. It takes three
# parameters:
# * `V` (pair potential),
# * `rho`: electronic density function
# * `embed` : embedding function
# """
# type EAMPotential{T1 <: PairPotential,
#                   T2 <: PairPotential,
#                   T3 <: Potential} <: SitePotential
#     V::T1
#     rho::T2
#     embed::T3
# end
#
# "embedding function for the Gupta potential"
# type GuptaEmbed <: Potential
#     xi
# end
# @inline evaluate(p::GuptaEmbed, r) = p.xi * sqrt(r)
# @inline evaluate_d(p::GuptaEmbed, r) = 0.5 * p.xi ./ sqrt(r)
#
# """
# `GuptaPotential`:
#     E_i = A ∑_{j ≠ i} v(r_ij) - ξ ∑_i √ ρ_i
#         v(r_ij) = exp[ -p (r_ij/r0 - 1) ]
#         ρ_i = ∑_{j ≠ i} exp[ -2q (r_ij / r0 - 1) ]
# """
# GuptaPotential(A, xi, p, q, r0, TC::Type, TCargs...)  =
#     EAMPotential( TC( SimpleExponential(A, p, r0), TCargs... ),      # V
#                   TC( SimpleExponential(1.0, 2*q, r0), TCargs...),   # rho
#                   GuptaEmbed( xi ) )                                 # embed
#
# "`EAMCalculator` : basic calculator using the `EAMPotential` type"
# type EAMCalculator <: AbstractCalculator
#     p::EAMPotential
# end
#
# cutoff(calc::EAMCalculator) = max(cutoff(calc.p.V), cutoff(calc.p.rho))
#
# function potential_energy(at::ASEAtoms, calc::EAMCalculator)
#     i, r = neighbour_list(at, "id", cutoff(calc))
#     return ( sum(calc.p.V(r))
#              + sum( calc.p.embed( simple_binsum( i, calc.p.rho(r) ) ) ) )
# end
#
# function potential_energy_d(at::ASEAtoms, calc::EAMCalculator)
#     i, j, r, R = neighbour_list(at, "ijdD", cutoff(calc))
#     # pair potential component
#     G = - 2.0 * simple_binsum(i, @GRAD calc.p.V(r, R'))
#     # EAM component
#     dF = @D calc.p.embed( simple_binsum(i, calc.p.rho(r)) )
#     dF_drho = dF[i]' .* (@GRAD calc.p.rho(r, R'))
#     G += simple_binsum(j, dF_drho) - simple_binsum(i, dF_drho)
#     return G
# end
