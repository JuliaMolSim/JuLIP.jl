
# JuLIP.jl master file.

module JuLIP

using Reexport

export Atoms

# TODO: correctly use import versus using throughout this package!

# quickly switch between Matrices and Vectors of Vecs
include("arrayconversions.jl")

# define types and abstractions of generic functions
include("abstractions.jl")

# a few auxiliary routines
include("utils.jl")

# implementation of some key functionality via ASE
include("ASE.jl")

# interface to DFT codes
include("DFT.jl")

# define the default atoms object
"""
`type Atoms`

Technically not a type but a type-alias, to possibly allow different "backends".
At the moment, `Atoms = ASE.ASEAtoms`; see its help for more details.
This will likely remain for the foreseeable future.
"""
typealias Atoms ASE.ASEAtoms

# only try to import Visualise, it is not needed for the rest to work.
try
   # some visualisation options
   include("Visualise.jl")
catch
   warn("""JuLIP.Visualise did not import correctly, probably because
               `imolecule` is not correctly installed.""")
end

# submodule JuLIP.Constraints
include("Constraints.jl")
@reexport using JuLIP.Constraints

# interatomic potentials prototypes and some example implementations
include("Potentials.jl")

# basic preconditioning capabilities
include("preconditioners.jl")

# some solvers
include("Solve.jl")
@reexport using JuLIP.Solve

# codes to facilitate testing
include("Testing.jl")


end # module
