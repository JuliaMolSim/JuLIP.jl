using JuLIP, ForwardDiff, StaticArrays, ReverseDiff, BenchmarkTools, Test

# simple EAM-like potential
f(R) = sqrt( 1.0 + sum( exp(-norm(r)) for r in R ) )

# hand-coded gradient
function f_d(R::JVecsF)
   ∇f = zeros(JVecF, length(R))
   ρ̄ = sum( exp(-norm(r)) for r in R )
   dF = 0.5 * (1.0 + ρ̄)^(-0.5)
   return [ dF * (- exp(-norm(r))) * r/norm(r)  for r in R ]
end

# ForwardDiff gradient
f_fd(R) =  ForwardDiff.gradient( T -> f(vecs(T)),  mat(R)[:] ) |> vecs

# turn it into an AD potential
const V = JuLIP.Potentials.ADPotential(f)

# a test configuration
nneigs = 30
R0 = [ @SVector rand(3) for n = 1:nneigs ]

# check that all of them are the same
@test V(R0) == f(R0)
@test (@D V(R0)) == f_fd(R0)
@test f_d(R0) ≈ f_fd(R0)

# timings
print("Timing for      f: ")
@btime f($R0);
print("Timing for      V: ")
@btime V($R0);
print("Timing for    f_d: ")
@btime f_d($R0);
print("Timing for   f_fd: ")
@btime f_fd($R0);
print("Timing for   @D V: ")
@btime (@D V($R0));
