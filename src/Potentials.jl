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

using JuLIP: AbstractAtoms, AbstractNeighbourList, AbstractCalculator,
      bonds, sites,
      JVec, JVecs, mat, vec

# we also import grad from JuLIP, but to define derivatives
import JuLIP: grad, energy, forces, cutoff

export Potential, PairPotential

"""
`Potential`: generic abstract supertype for all potential-like things
"""
abstract Potential <: AbstractCalculator


"""
`PairPotential`: abstract supertype for pair potentials
"""
abstract PairPotential <: Potential


include("potentials_base.jl")
# * @pot, @D, @DD, @GRAD and related things


include("analyticpotential.jl")
# * AnalyticPotential

include("cutoffs.jl")
#   * SWCutoff
#   * ShiftCutoff
#   * SplineCutoff

include("pairpotentials.jl")
# * PairCalculator
# * LennardJonesPotential
# * MorsePotential
# * SimpleExponential

include("EMT.jl")
# * EMTCalculator

include("stillingerweber.jl")
# * type StillingerWeber



export ZeroSitePotential

@pot type ZeroSitePotential <: Potential
end

evaluate(p::ZeroSitePotential, r, R) = 0.0
evaluate_d(p::ZeroSitePotential, r, R) = zeros(r)



end
