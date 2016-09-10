
# test codes for
#   * matrix vec conversions

X = rand(3, 3)
Y = vecs(X) |> mat
@assert X == Y
Y[1] = -1.0
@assert X == Y
