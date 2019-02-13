
h1("Test Aux")
h2("matrix <-> vec conversions")
X = rand(3, 3)
Y = vecs(X) |> mat
println(@test X == Y)
Y[1] = -1.0
println(@test X == Y)

V = rand(JVecF, 10)
A = mat(V)
B = vecs(A)
println(@test B == V)
B[1] = rand(JVecF)
println(@test B == V)
