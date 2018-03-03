
# JuLIP.jl master file.

module JuLIP

using Reexport, NeighbourLists, StaticArrays, Parameters


# quickly switch between Matrices and Vectors of Vecs
include("arrayconversions.jl")

# define types and abstractions of generic functions
include("abstractions.jl")

# implementation of some key functionality via ASE
include("ASE.jl")

include("chemistry.jl")

# the main atoms type
include("atoms.jl")


# a few auxiliary routines
include("utils.jl")

# # only try to import Visualise, it is not needed for the rest to work.
# try
#    # some visualisation options
#    if isdefined(Main, :JULIPVISUALISE)
#       if Main.JULIPVISUALISE == true
#          include("Visualise.jl")
#       end
#    end
# catch
#    JuLIP.julipwarn("""JuLIP.Visualise did not import correctly, probably because
#                `imolecule` is not correctly installed.""")
# end
#
# # interatomic potentials prototypes and some example implementations
# include("Potentials.jl")

# # submodule JuLIP.Constraints
# include("Constraints.jl")
# @reexport using JuLIP.Constraints
#
# # basic preconditioning capabilities
# include("preconditioners.jl")
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

end # module
