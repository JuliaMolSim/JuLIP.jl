

# test codes for
#   * matrix vec conversions

X = rand(3, 3)
Y = vecs(X) |> mat
@test X == Y
Y[1] = -1.0
@test X == Y

# the positions test is also a test for the py-reference thing

# check the positions-as-ref thing
at = Atoms("Al") * (2,3,2)
X = unsafe_positions(at)
X[1] = JVec(rand(3))
@test X == positions(at)

# test set_positions!
Y = copy(X)
Y[3] = JVec(rand(3))
@test Y != positions(at)
set_positions!(at, Y)
@test Y == positions(at)
