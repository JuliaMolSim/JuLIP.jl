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
    @test ustrip.(u"Å/s", velocity(ab,i)) ≈ at.P[i]
end