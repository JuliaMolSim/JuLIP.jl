
# JuLIP.jl master file.

module JuLIP

# warn("""This is a major breaking change for JuLIP, effectively removing
#         the dependency on ASE. If you have been using the JuLIP master branch
#         until now, then please checkout the branch `v0.2ase`. Access to ASE
#         will be restored asap, in a separate package. Documentation
#         on how to use the 'new JuLIP' will hopefully follow very soon.""")

using Reexport, NeighbourLists, StaticArrays, Parameters


# quickly switch between Matrices and Vectors of SVectors, etc
include("arrayconversions.jl")

# define types and abstractions of generic functions
include("abstractions.jl")

include("chemistry.jl")
@reexport using JuLIP.Chemistry

# the main atoms type
include("atoms.jl")

# File IO
include("FIO.jl")
@reexport using JuLIP.FIO

# how to build some simple domains
include("build.jl")
@reexport using JuLIP.Build

# a few auxiliary routines
include("utils.jl")

# interatomic potentials prototypes and some example implementations
include("Potentials.jl")
@reexport using JuLIP.Potentials

# submodule JuLIP.Constraints
include("Constraints.jl")
@reexport using JuLIP.Constraints

# basic preconditioning capabilities
include("preconditioners.jl")
@reexport using JuLIP.Preconditioners

# some solvers
include("Solve.jl")
@reexport using JuLIP.Solve

# experimental features
include("Experimental.jl")
@reexport using JuLIP.Experimental

# codes to facilitate testing
include("Testing.jl")


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


end # module
