

using JuLIP
using Test
using LinearAlgebra: I

h3("check that `bulk` evaluates ok...")
at = bulk(:Si)
println(@test typeof(at) == Atoms{Float64})

h3("... and that we can repeat it.")
println(@test length(at * (1,2,3)) == 6 * length(at))

h3("check deepcopy and == ...")
at = bulk(:Si)
at1 = deepcopy(at)
println(@test at == at1)

h3("Check setindex! and getindex ...")
at = bulk(:Si, cubic=true)
x = at[2]
println(@test x == JVec(1.3575, 1.3575, 1.3575))
x = x .+ 0.1
at[2] = x
X = positions(at)
println(@test !(X === at.X))
println(@test X[2] == x)


h3("set_positions ...")
at = bulk(:Si, cubic=true) * 2
X = positions(at)
println(@test !(X === at.X))
println(@test X == at.X)
X += 0.1 * rand(JVecF, length(at))
println(@test X != at.X)
set_positions!(at, X)
println(@test X == at.X)
println(@test !(X === at.X))

# TODO: set_momenta, set_masses, set_numbers

println(@test chemical_symbols(at) == fill(:Si, 64))
println(@test chemical_symbols(bulk(:W)) == [:W])



h3("test set_positions!")
Y = copy(X)
Y[3] = JVec(rand(3))
println(@test Y != positions(at))
set_positions!(at, Y)
println(@test Y == positions(at))

h3("test set_momenta!")
Nat = length(at)
Ifree = rand(collect(1:Nat), Nat * 4 ÷ 5) |> unique |> sort # prepare for test below
Iclamp = setdiff(1:Nat, Ifree)
P = rand(size(Y |> mat)...)
P[:, Iclamp] .= 0.0
P = vecs(P)
set_momenta!(at, P)
println(@test P == momenta(at))


h3("test set_dofs!, etc")
# this is making an assumptions on the ordering of dofs; since a new
# implementation of the DOF manager could change this, this test needs to be
# re-implemented if that happens.
set_free!(at, Ifree)
println(@test dofs(at) == position_dofs(at) == mat(Y)[:, Ifree][:])
println(@test momentum_dofs(at) == mat(P)[:, Ifree][:])
q = position_dofs(at)
q = rand(length(q))
set_position_dofs!(at, q)
println(@test q == position_dofs(at))
p = rand(length(q))
set_momentum_dofs!(at, p)
println(@test p == momentum_dofs(at))
