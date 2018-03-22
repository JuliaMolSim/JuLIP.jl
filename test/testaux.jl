

println("test matrix <-> vec conversions")
X = rand(3, 3)
Y = vecs(X) |> mat
@test X == Y
Y[1] = -1.0
@test X == Y
