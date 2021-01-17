using Test
using ASE
using JuLIP

eam_alloy = "data/PdAgH_HybridPd3Ag.eam.alloy"
eam_fs = "data/Fe-P.eam.fs"
@test JuLIP.Potentials.EAM(eam_alloy) isa EAM{T, 1} where {T}
@test JuLIP.Potentials.EAM(eam_fs) isa EAM{T, 2} where {T}

eam_Fe = JuLIP.Potentials.EAM1("data/pfe.plt", "data/ffe.plt", "data/F_fe.plt")
eam_Fe2 = JuLIP.Potentials.EAM("data/pfe.plt", "data/ffe.plt", "data/F_fe.plt")

at = rattle!(bulk(:Fe) * 4, 0.5)
@test energy(eam_Fe, at) ≈ energy(eam_Fe2, at)
@test forces(eam_Fe, at) ≈ forces(eam_Fe2, at)
