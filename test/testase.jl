
println("-------------------")
println(" Testing JuLIP.ASE")
println("-------------------")

using JuLIP
using JuLIP.ASE

# ======================================================================

println("Check that the cubic unit cell for Al is reproduced correctly")
a0 = 2.025
p = [ [0.0;0.0;0.0] [0.0;a0;a0] [a0;0.0;a0] [a0;a0;0.0] ]
at = bulk("Al", cubic=true)
@test (positions(at) |> mat) == p

# ======================================================================

println("Check neighbourlist without periodicity")
# TODO: implement a test with periodicity!!!
println("   TODO: implement test with periodicity as well?")
println("   ... assemble neighbour list ...")
at = bulk("Al", cubic=true) * 3
set_pbc!(at, (false,false,false))
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

println("Checking `***_data`, `***_array`, `***_info`, `***_transient`")

at = bulk("Cu") * 3
N = length(at)
# set an array and test that it is read back correctly
z = rand(N)
set_array!(at, "z", z)
@test get_array(at, "z") == z
@test get_data(at, "z") == z
# set data and check it is read as an array
set_data!(at, "y", z)
@test get_array(at, "y") == z
# set some info and test reading
i = "some info"
set_info!(at, "i", i)
@test get_info(at, "i") == i
@test get_data(at, "i") == i
# Now try to set an array "i" and check that we get an error
caught = false
try
   set_array!(at, "i", z)
catch
   caught = true
end
@test caught
# ***_transient should be tested automatically via the calculators.
# test the has_***
@test has_data(at, "z")
@test has_data(at, "i")
@test has_array(at, "z")
@test has_info(at, "i")
@test !has_array(at, "i")
@test !has_info(at, "z")


println("Checking momenta and velocites")
p = rand(3,N) |> vecs
set_momenta!(at, p)
@test_approx_eq momenta(at) p
v = p ./ masses(at)
@test_approx_eq velocities(at) v
set_velocities!(at, v/2.0)
@test_approx_eq velocities(at) v/2.0
@test_approx_eq momenta(at) p/2.0

println("Checking chemical_symbols")
@test chemical_symbols(at) == ["Cu" for i in 1:length(at)]
