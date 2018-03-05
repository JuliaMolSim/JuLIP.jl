

using JuLIP
using Base.Test


println("check that `bulk` evaluates ok...")
at = bulk(:Si)
@test typeof(at) == Atoms{Float64, Int64}

println("... and that we can repeat it.")
@test length(at * (1,2,3)) == 6 * length(at)

println("check deepcopy and == ...")
at = bulk(:Si)
at1 = deepcopy(at)
@test at == at1

if hasase
   println("Check correct implementation of `repeat` and `*` ...")
   for n in [ (2,1,1), (2,2,1), (2,3,4), (2,3,1) ]
      @test (at * n) == Atoms(ASE.bulk(:Si) * n) == repeat(at, n)
   end

   println("   check correct repeat of momenta ...")
   at_ase = ASE.bulk("Si")
   P = rand(JVecF, 2)
   set_momenta!(at_ase, P)
   set_momenta!(at, P)
   @test Atoms(at_ase) == at
   @test Atoms(at_ase * (2,4,3)) == (at * (2,4,3))
end 

println("Check setindex! and getindex ...")
at = bulk(:Si, cubic=true)
x = at[2]
@test x == JVec(1.3575, 1.3575, 1.3575)
x += 0.1
at[2] = x
X = positions(at)
@test !(X === at.X)
@test X[2] == x


println("set_positions ...")
at = bulk(:Si, cubic=true) * 2
X = positions(at)
@test !(X === at.X)
@test X == at.X
X += 0.1 * rand(JVecF, length(at))
@test X != at.X
set_positions!(at, X)
@test X == at.X
@test !(X === at.X)

# TODO: set_momenta, set_masses, set_numbers

@test chemical_symbols(at) == fill(:Si, 64)
@test chemical_symbols(bulk(:W)) == [:W]



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
