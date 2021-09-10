# JuLIP.jl master file.

module JuLIP

using Reexport
@reexport using NeighbourLists

const _usethreads = Ref(true)
function usethreads!(tf::Bool)
   JuLIP._usethreads[] = tf
end
nthreads() = JuLIP._usethreads[] ? Threads.nthreads() : 1

# quickly switch between Matrices and Vectors of SVectors, etc
include("arrayconversions.jl")

import ACEbase: FIO
@reexport using JuLIP.FIO

# define types and abstractions of generic functions
include("abstractions.jl")

include("chemistry.jl")
@reexport using JuLIP.Chemistry

# the main atoms type
include("atoms.jl")
include("dofmanagement.jl")

# how to build some simple domains
include("build.jl")
@reexport using JuLIP.Build

# a few auxiliary routines
include("utils.jl")
@reexport using JuLIP.Utils

# interatomic potentials prototypes and some example implementations
include("Potentials.jl")
@reexport using JuLIP.Potentials
# and we want to import some more functions from `Potentials` which are really
# central to JuLIP, so that they can be extended using just `import JuLIP: ...`
import JuLIP.Potentials: numz, z2i, i2z

# basic preconditioning capabilities
include("preconditioners.jl")
@reexport using JuLIP.Preconditioners

# some solvers
include("Solve.jl")
@reexport using JuLIP.Solve

# experimental features
include("Experimental.jl")
@reexport using JuLIP.Experimental

# the following are some sub-modules that are primarily used
# to create further abstractions to be shared across several
# modules in the JuLIP-verse.
include("mlips.jl")


# codes to facilitate testing
include("Testing.jl")
include("datadeps.jl")

end # module
