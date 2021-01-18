using Test
using JuLIP
using BenchmarkTools

@testset "Compare EAM and EAM1 (No ASE)" begin
    fs = "data/Au.fs"
    @testset "Convert from EAM to EAM1" begin
        fs_model = EAM(fs)
        at = rattle!(bulk(:Au) * (5, 5, 4), 0.5)
        @test energy(fs_model, at) ≈ energy(EAM1(fs_model), at)
        @test forces(fs_model, at) ≈ forces(EAM1(fs_model), at)
        @test hessian(fs_model, at) ≈ hessian(EAM1(fs_model), at)
    end

    @testset "Test fs constructor change" begin # eam1_from_fs is now redundant
        eam = Potentials.eam_from_fs(fs)
        eam1 = Potentials.eam1_from_fs(fs)
        at = rattle!(bulk(:Au) * (5, 5, 4), 0.5)
        @test energy(eam, at) ≈ energy(eam1, at)
        @test forces(eam, at) ≈ forces(eam1, at)
        @test hessian(eam, at) ≈ hessian(eam1, at)
    end

    @testset "Old and new constructor" begin # Can remove this EAM1 constructor
        eam1_Fe = EAM1("data/pfe.plt", "data/ffe.plt", "data/F_fe.plt")
        eam_Fe = EAM("data/pfe.plt", "data/ffe.plt", "data/F_fe.plt")
        at = rattle!(bulk(:Fe) * 4, 0.5)
        @test energy(eam1_Fe, at) ≈ energy(eam_Fe, at)
        @test forces(eam1_Fe, at) ≈ forces(eam_Fe, at)
        @test hessian(eam1_Fe, at) ≈ hessian(eam_Fe, at)
    end

    @testset "Performance" begin
        at = rattle!(bulk(:Au) * (5, 5, 4), 0.5)
        fs_model = EAM(fs)
        fs_model1 = EAM1(fs_model)
        @btime energy($fs_model, $at)
        @btime energy($fs_model1, $at)
        @btime forces($fs_model, $at)
        @btime forces($fs_model1, $at)
        @btime hessian($fs_model, $at)
        @btime hessian($fs_model1, $at)
        #=
        342.200 μs (4 allocations: 1.06 KiB)
        350.399 μs (4 allocations: 1.06 KiB)
        543.999 μs (6 allocations: 4.44 KiB)
        542.400 μs (6 allocations: 4.44 KiB)
        40.810 ms (8272 allocations: 80.21 MiB)
        40.623 ms (8272 allocations: 80.21 MiB)
        =#
    end
end

@testset "EAM with ASE" begin
    using ASE
    alloy = "data/PdAgH_HybridPd3Ag.eam.alloy"
    eam_fs = "data/Fe-P.eam.fs"
    fs = "data/Au.fs"
    @test EAM(alloy) isa EAM{T, 1} where {T}
    @test EAM(eam_fs) isa EAM{T, 2} where {T}
    @test EAM(fs) isa EAM{T, 2} where {T}
end
