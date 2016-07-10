
println("-------------------")
println(" Testing JuLIP.ASE")
println("-------------------")

using JuLIP

# ======================================================================

println("Check that the cubic unit cell for Al is reproduced correctly")
a0 = 2.025
p = [ [0.0;0.0;0.0] [0.0;a0;a0] [a0;0.0;a0] [a0;a0;0.0] ]
at = Atoms("Al", cubic=true)
@test (positions(at) |> mat) == p

# ======================================================================

println("Check neighbourlist without periodicity")
# TODO: implement a test with periodicity!!!
println("   TODO: implement test with periodicity as well?")
println("   ... assemble neighbour list ...")
at = Atoms("Al", cubic=true, pbc = (false,false,false), repeatcell=(3,3,3))
cutoff = 1.7 * a0
nlist = neighbourlist(at, cutoff)
# create a neighbourlist via a naive double-loop
simple = zeros(length(at), length(at))
X = positions(at)
for n = 2:length(at), m = 1:n-1
   if norm(X[m]-X[n]) <= cutoff
      simple[n,m] = simple[m,n] = 1
   end
end
println("   ... check the bond-iterator ... ")
for (i,j,r,R,S) in bonds(nlist)
   @test simple[i,j] == 1
   @test_approx_eq_eps(norm(X[i]-X[j]), r, 1e-12)
   @test_approx_eq_eps(X[j]-X[i], R, 1e-12)
   @test norm(S) == 0
   # switch the flag
   simple[i,j] = -1
end
# check that all pairs have been found
@test maximum(simple) == 0
# revert to original
simple *= -1
println("   ... check the site iterator ... ")
for (i,j,r,R,S) in sites(nlist)
   for n = 1:length(j)
      @test simple[i,j[n]] == 1
      simple[i,j[n]] = -1
   end
   @test maximum(simple[i,:]) == 0
end
@test maximum(simple) == 0

# ======================================================================
