

using JuLIP
using Test
using LinearAlgebra: I 

h3("check that `bulk` evaluates ok...")
at = bulk(:Si)
println(@test typeof(at) == Atoms{Float64, Int64})

h3("... and that we can repeat it.")
println(@test length(at * (1,2,3)) == 6 * length(at))

h3("check deepcopy and == ...")
at = bulk(:Si)
at1 = deepcopy(at)
println(@test at == at1)

if hasase
   h3("Check correct implementation of `repeat` and `*` ...")
   for n in [ (2,1,1), (2,2,1), (2,3,4), (2,3,1) ]
      println(@test (at * n) == Atoms(ASE.bulk(:Si) * n) == repeat(at, n))
   end

   h3("   check correct repeat of momenta ...")
   at_ase = ASE.bulk("Si")
   P = rand(JVecF, 2)
   set_momenta!(at_ase, P)
   set_momenta!(at, P)
   println(@test Atoms(at_ase) == at)
   println(@test Atoms(at_ase * (2,4,3)) == (at * (2,4,3)))
end

h3("Check setindex! and getindex ...")
at = bulk(:Si, cubic=true)
x = at[2]
println(@test x == JVec(1.3575, 1.3575, 1.3575))
x += 0.1
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



# println("test set_positions!")
# Y = copy(X)
# Y[3] = JVec(rand(3))
# @test Y != positions(at)
# set_positions!(at, Y)
# @test Y == positions(at)
#
# println("test set_momenta!")
# Nat = length(at)
# Ifree = rand(collect(1:Nat), Nat * 4 ÷ 5) |> unique |> sort # prepare for test below
# Iclamp = setdiff(1:Nat, Ifree)
# P = rand(size(Y |> mat)); P[:, Iclamp] = 0.0; P = vecs(P)
# set_momenta!(at, P)
# @test P == momenta(at)
#
#
# println("test set_dofs!, etc")
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
