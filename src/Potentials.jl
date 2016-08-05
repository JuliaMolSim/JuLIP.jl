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


export LennardJonesPotential, MorsePotential, AnalyticPotential,
   SWCutoff, ShiftCutoff, SplineCutoff,
   LennardJonesCalculator, MorseCalculator, GuptaCalculator,
   @D, @DD, @GRAD

"""
`Potential`: generic abstract supertype for all potential-like things
"""
abstract Potential

# default method to print potentials:
Base.show(io::Base.IO, p::Potential) = print(io, string(p))
Base.print(io::Base.IO, p::Potential) = print(io, string(p))



"""
`PairPotential`: abstract supertype for pair potentials
"""
abstract PairPotential <: Potential


include("potentials_base.jl")
# *
#
#


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


################### the EMT Calculator ###################

# include("emt.jl")



end
