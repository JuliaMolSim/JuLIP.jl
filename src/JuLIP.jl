
# JuLIP.jl master file.

module JuLIP

using Reexport, NeighbourLists, StaticArrays, Parameters


# quickly switch between Matrices and Vectors of SVectors, etc
include("arrayconversions.jl")

# define types and abstractions of generic functions
include("abstractions.jl")

include("chemistry.jl")
@reexport using JuLIP.Chemistry

# # the main atoms type
# include("atoms.jl")
#
# # File IO
# include("FIO.jl")
# @reexport using JuLIP.FIO
#
# # how to build some simple domains
# include("build.jl")
# @reexport using JuLIP.Build
#
# # a few auxiliary routines
# include("utils.jl")
#
# # interatomic potentials prototypes and some example implementations
# include("Potentials.jl")
# @reexport using JuLIP.Potentials
#
# # submodule JuLIP.Constraints
# include("Constraints.jl")
# @reexport using JuLIP.Constraints
#
# # basic preconditioning capabilities
# include("preconditioners.jl")
# @reexport using JuLIP.Preconditioners
#
# # some solvers
# include("Solve.jl")
# @reexport using JuLIP.Solve
#
# # experimental features
# include("Experimental.jl")
# @reexport using JuLIP.Experimental
#
# # codes to facilitate testing
# include("Testing.jl")
#
#
# # the following are some sub-modules that are primarily used
# # to create further abstractions to be shared across several
# # modules in the JuLIP-verse.
# include("mlips.jl")
# include("nbody.jl")


end # module
