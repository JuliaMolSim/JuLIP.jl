
using Test, JuLIP, LinearAlgebra
using JuLIP.Potentials
using JuLIP.Testing

##
@info("Testing single-species One-Body")
E0 = rand()
V = OneBody(:Si => E0)
at = bulk(:Si, cubic=true) * (2,1,2)
println(@test energy(V, at) ≈ length(at) * E0)
println(@test norm(forces(V, at)) == 0)
D = write_dict(V)
V1 = JuLIP.read_dict(D)
println(@test V1 == V)

@info("Testing multi-species One-Body")
E1 = rand()
V2 = OneBody(:Si => E0, :C => E1)
at.Z[2:2:end] .= atomic_number(:C)
println(@test energy(V2, at) ≈ length(at) * (E0+E1) / 2)
println(@test norm(forces(V, at)) == 0)
D = write_dict(V2)
V21 = JuLIP.read_dict(D)
println(@test V2 == V21)
