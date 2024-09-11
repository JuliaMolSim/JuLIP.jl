using AtomsBase
using JuLIP
using Test
using Unitful


##
at = bulk(:Si)*2
at.P[1] = rand(3)

ab = FlexibleSystem(at)
abc = convert(AbstractSystem, at)
atc = convert(Atoms, ab)
att = Atoms(ab)

@test at.X ≈ att.X
@test at.P ≈ att.P
@test at.cell ≈ att.cell
@test at.M ≈ att.M

@test all( ab[:bounding_box] .≈ abc[:bounding_box] )
@test at.cell ≈ att.cell
map( 1:3 ) do i
    @test ustrip.(u"Å", ab[:bounding_box][i]) ≈ at.cell[i,:]
end

# Test position conversion
map( 1:length(ab) ) do i
    @test ustrip.(u"Å", position(ab,i)) ≈ at.X[i]
end

# Test velocities
map( 1:length(ab) ) do i
    @test ustrip.(u"eV^0.5/u^0.5", velocity(ab,i)) ≈ at.P[i] / at.M[i]
end

## Test conversion of data parameters
hydrogen = isolated_system([:H => [0, 0, 1.]u"Å",
                            :H => [0, 0, 3.]u"Å"];
                            e=1.3)
a = Atoms(hydrogen)

@test a.data["e"].data == hydrogen[:e]

hh = FlexibleSystem(a)

@test hh[:e] == hydrogen[:e]

## Test non cubic cell

a1 = AtomsBase.Atom( :Ti, [1.4175416,1.3076719,2.0027419]u"Å" )
a2 = AtomsBase.Atom( :Al, [-0.0784151, -0.17961, -0.0164296]u"Å" )
cell = ( [2.65283,  0.0,      0.0]u"Å", [0.0,      2.65283,  0.0273594]u"Å", [0.0,      0.0,      3.82503]u"Å" )
pbc = (true, true, true)

ab = FlexibleSystem([a1,a2], cell, pbc)

aj = JuLIP.Atoms(ab)

ab2 = FlexibleSystem(aj)

@test all( bounding_box(ab) .≈ bounding_box(ab2) )
@test bounding_box(ab)[2][3] ≈ aj.cell[2,3] * u"Å"
@test bounding_box(ab)[3][2] ≈ aj.cell[3,2] * u"Å"