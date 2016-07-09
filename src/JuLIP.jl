# JuLIP.jl master file.
#



module JuLIP

export Atoms


# quickly switch between Matrices and Vectors of Vecs
include("arrayconversions.jl")

# define types and abstractions of generic functions
include("abstractions.jl")

# implementation of some key functionality via ASE
# include("ASE.jl")


# # define aliases!
# """
# `type Atoms`
#
# Technically not a type but a type-alias, to allow different "backends".
# At the moment, `Atoms = ASE.ASEAtoms`.
# """
# Atoms = ASE.ASEAtoms



end # module
