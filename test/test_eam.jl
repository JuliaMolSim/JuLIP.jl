using Test
using JuLIP

data = joinpath(dirname(pathof(JuLIP)), "..", "data") * "/"

@testset "EAM with ASE" begin
    using ASE

    alloy = data * "PdAgH_HybridPd3Ag.eam.alloy"
    eam_fs = data * "Fe-P.eam.fs"
    Ni = data * "Ni.eam.fs"

    @test EAM(alloy) isa EAM{T, 1} where {T}
    @test EAM(eam_fs) isa EAM{T, 2} where {T}

    eam = pyimport("ase.calculators.eam")
    ase_calc = ASECalculator(eam.EAM(potential=Ni))
    # make sure we get a perfect fit ...
    julip_calc = EAM(Ni)

    atoms = bulk(:Ni) * 3
    rattle!(atoms, 0.1)
    E_jl = energy(julip_calc, atoms)
    E_py = energy(ase_calc, atoms)
    # ... but even then we the evaluation codes aren't the
    #     same so we only get ca 1e-6 to 1e-7 match.
    @test abs(E_jl - E_py) < 1e-5

    # Test again with unordered species in parameter file
    ase_calc = ASECalculator(eam.EAM(potential=alloy))
    julip_calc = EAM(alloy)

    atoms = bulk(:Pd) * 3
    rattle!(atoms, 0.1)
    E_jl = energy(julip_calc, atoms)
    E_py = energy(ase_calc, atoms)
    @test E_jl ≈ E_py rtol=1e-3
end
