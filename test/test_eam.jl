using Test
using ASE
using LinearAlgebra
import Random
using JuLIP
using DataDeps
using Optim

data = joinpath(dirname(pathof(JuLIP)), "..", "data") * "/"

##
# @testset "EAM with ASE" begin

test_pots = joinpath(datadep"JuLIP_testpots", "JuLIP_data") * "/"
alloy = test_pots * "PdAgH_HybridPd3Ag.eam.alloy"
eam_fs = test_pots * "Fe-P.eam.fs"
Ni = test_pots * "Ni.eam.fs"

@test EAM(alloy) isa EAM{T, 1} where {T}
@test EAM(eam_fs) isa EAM{T, 2} where {T}

eam = pyimport("ase.calculators.eam")
ase_calc = ASECalculator(eam.EAM(potential=Ni))
# make sure we get a perfect fit ...
julip_calc = EAM(Ni)

atoms = bulk(:Ni) * 3
Random.seed!(0)
rattle!(atoms, 0.1)
E_jl = energy(julip_calc, atoms)
E_py = energy(ase_calc, atoms)
# ... but even then we the evaluation codes aren't the
#     same so we only get ca 1e-6 to 1e-7 match.
@test abs(E_jl - E_py) < 1e-10

# Test again with unordered species in parameter file
ase_calc = ASECalculator(eam.EAM(potential=alloy))
julip_calc = EAM(alloy)

atoms = bulk(:Pd) * 3
atoms.Z[1:3:end] .= AtomicNumber(:Ag)
atoms.Z[1:5:end] .= AtomicNumber(:H)
rattle!(atoms, 0.1)
E_jl = energy(julip_calc, atoms)
E_py = energy(ase_calc, atoms)
@test E_jl ≈ E_py rtol=1e-6
# end

# Test a few quantities of interest with respect to reference values obtained with LAMMPS
# You can set `got_lammps`to true if you have ASE and LAMMPS Python libraries installed

got_lammps = false
pot_file = test_pots * "w_eam4.fs"
eam_julip = EAM(pot_file)

if got_lammps
    lammpslib = pyimport("ase.calculators.lammpslib")
    eam_lammps = ASECalculator(lammpslib.LAMMPSlib(lmpcmds=["pair_style eam/fs",
                                "pair_coeff * * $pot_file W"],
                                atom_types=Dict("W" => 1), keep_alive=true))
end

# simple energy test
unitcell = bulk(:W, cubic=true)
@test energy(eam_julip, unitcell) ≈ -17.792788114962622
if got_lammps
    @show energy(eam_julip, unitcell)
    @show energy(eam_lammps, unitcell)
    @test energy(eam_julip, unitcell) ≈ energy(eam_lammps, unitcell)
end

# lattice constant
eam = eam_julip # or eam_lammps
variablecell!(unitcell)
set_calculator!(unitcell, eam)
res = optimize(x -> energy(unitcell, x), x -> gradient(unitcell, x), 
                dofs(unitcell), inplace=false)
alat = unitcell.cell[1, 1]
@test alat ≈ 3.1433900286308045

# (110) surface energy
shift = 2.0
ase_lattice_cubic = pyimport("ase.lattice.cubic")
make_bulk() = Atoms(ASEAtoms(ase_lattice_cubic.BodyCenteredCubic(symbol="W", 
                             latticeconstant=alat, 
                             directions=[[1,-1,0],[1,1,0],[0,0,1]]) * (1, 1, 10)))
bulk_at = make_bulk()                             
set_calculator!(bulk_at, eam)
surface = make_bulk() #copy(bulk_at) # is there a better way to do this in JuLIP?
X = positions(surface) |> mat
X[3, :] .+= shift
set_positions!(surface, X)
wrap_pbc!(surface)
c = Matrix(surface.cell)
c[3, :] += [0.0, 0.0, 10.0]
set_cell!(surface, c)
fixedcell!(surface)
set_calculator!(surface, eam)
area = norm(cross(bulk_at.cell[:, 1], bulk_at.cell[:, 2]))
γ = (energy(surface) - energy(bulk_at)) / (2 * area)
@show γ
@test γ ≈ 0.18415972172748984