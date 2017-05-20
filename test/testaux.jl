

# test codes for
#   * matrix vec conversions

X = rand(3, 3)
Y = vecs(X) |> mat
@test X == Y
Y[1] = -1.0
@test X == Y

# the positions test is also a test for the py-reference thing

# check the positions-as-ref thing
at = bulk("Al") * (2,3,2)
X = unsafe_positions(at)
X[1] = JVec(rand(3))
@test X == positions(at)

# test set_positions!
Y = copy(X)
Y[3] = JVec(rand(3))
@test Y != positions(at)
set_positions!(at, Y)
@test Y == positions(at)

# test set_momenta!
Nat = length(at)
Ifree = rand(collect(1:Nat), Nat * 4 ÷ 5) |> unique |> sort # prepare for test below
Iclamp = setdiff(1:Nat, Ifree)
P = rand(size(Y |> mat)); P[:, Iclamp] = 0.0; P = vecs(P)
set_momenta!(at, P)
@test P == momenta(at)


# TODO: this test failed; somehow we completely lost the momentum_dofs and position_dofs ???
# # test set_dofs, etc
# # this is making an assumptions on the ordering of dofs; since a new
# # implementation of the DOF manager could change this, this test needs to be
# # re-implemented if that happens.
# set_constraint!(at, FixedCell(at, free = Ifree))
# @test dofs(at) == position_dofs(at) == mat(Y)[:, Ifree][:]
# @test momentum_dofs(at) == mat(P)[:, Ifree][:]
# q = position_dofs(at)
# q = rand(length(q))
# set_position_dofs!(at, q)
# @test q == position_dofs(at)
# p = momentum_dofs(at)
# p = rand(length(p))
# set_momentum_dofs!(at, p)
# @test p == momentum_dofs(at)
