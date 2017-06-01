

println("test matrix <-> vec conversions")
X = rand(3, 3)
Y = vecs(X) |> mat
@test X == Y
Y[1] = -1.0
@test X == Y

# the positions test is also a test for the py-reference thing

println("check the positions-as-ref thing (and the pyref)")
at = bulk("Al") * (2,3,2)
X = unsafe_positions(at)
X[1] = JVec(rand(3))
@test X == positions(at)

println("test set_positions!")
Y = copy(X)
Y[3] = JVec(rand(3))
@test Y != positions(at)
set_positions!(at, Y)
@test Y == positions(at)

println("test set_momenta!")
Nat = length(at)
Ifree = rand(collect(1:Nat), Nat * 4 ÷ 5) |> unique |> sort # prepare for test below
Iclamp = setdiff(1:Nat, Ifree)
P = rand(size(Y |> mat)); P[:, Iclamp] = 0.0; P = vecs(P)
set_momenta!(at, P)
@test P == momenta(at)


println("test set_dofs!, etc")
# this is making an assumptions on the ordering of dofs; since a new
# implementation of the DOF manager could change this, this test needs to be
# re-implemented if that happens.
set_constraint!(at, FixedCell(at, free = Ifree))
@test dofs(at) == position_dofs(at) == mat(Y)[:, Ifree][:]
@test momentum_dofs(at) == mat(P)[:, Ifree][:]
q = position_dofs(at)
q = rand(length(q))
set_position_dofs!(at, q)
@test q == position_dofs(at)
p = momentum_dofs(at)
p = rand(length(p))
set_momentum_dofs!(at, p)
@test p == momentum_dofs(at)


println("testing periodic notion of distance")
X1 = [ JVecF(0.0,0.0,0.0), JVecF(0.5,0.5,0.5)]
at = ASEAtoms("H2", X1)
set_pbc!(at, (true, false, false))
set_cell!(at, [1.0 0.2 0.3; 0.0 1.0 0.1; 0.0 0.0 1.0])
X2 = [ X1[1], X1[2] + JVecF(1.0, 0.2, 0.3)]
X3 = [ X1[1] - JVecF(0.1, 0.0, 0.0), X1[2]]
X4 = [ X1[1], X1[2] + JVecF(0.0, 1.0, 0.1)]

@testset "dist" begin
   @test JuLIP.dist(at, X1, X2) < 1e-14
   @test JuLIP.dist(at, X1, X3) ≈ 0.1
   @test JuLIP.dist(at, X1, X4) ≈ norm(JVecF(0.0, 1.0, 0.1))
end
