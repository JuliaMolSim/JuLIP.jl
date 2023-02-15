# This is an example on how to use AtomsBase
using JuLIP
using AtomsBase
using Unitful

# Convert from JuLIP Atoms to AtomsBase
at = bulk(:Si)*2
ab = convert(AbstractSystem, at)
# also direct build works
ab_too = FlexibleSystem(at)

# Convert back to JuLIP Atoms
jat = Atoms(ab)


# Extra paremeters in AtomsBase
hydrogen = isolated_system([:H => [0, 0, 1.]u"Å",
                            :H => [0, 0, 3.]u"Å"];
                            energy = rand(1),
                            forces = rand(2,3),
                            )
H2 = Atoms(hydrogen)
# JuLIP.Atoms H2 contains now energy and forces information
# that can be used to train ACE models.
# The same way you can add virials and weights to
# AtomsBase structure. Which allows fine tuning training ACE models.

# To use AtomsBase structures where you would use JuLIP.Atoms
# you need to first convert to JuLIP.Atoms

# To convert individual structure call Atoms
Atoms(hydrogen)

# For a trajectory, which is a array of individual parts use broadcasting
H2_traj = Atoms.([hydrogen, hydrogen])


## AtomsIO allows loading structure from wide variety of files
using AtomsIO
# load individual structure
struc = load_system("path to some file")
# load a trajectory
traj = load_trajectory("trajectory file")

# Convert to JuLIP Atoms
a_struct = Atoms(struc)
a_traj = Atoms.(traj)

# You can then use the trajectory to train e.g. ACE models

# Note! As of this writing AtomsIO pulls PythonCall
# and you cannot use it together with PyCall that is
# needed to fit ACE models. So, you cannot use AtomsIO
# to load training data for ACE.