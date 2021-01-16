using Test
using JuLIP

filename = "data/PdAgH_HybridPd3Ag.eam.alloy"
@test JuLIP.Potentials.generic_EAM(filename) isa EAM

