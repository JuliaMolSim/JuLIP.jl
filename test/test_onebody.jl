
using Test, JuLIP, LinearAlgebra
using JuLIP.Potentials
using JuLIP.Testing

@info("Testing One-Body")
E0 = rand()
V = OneBody(E0)
at = bulk(:Si, cubic=true) * (2,1,2)
println(@test energy(V, at) ≈ length(at) * E0)
println(@test norm(forces(V, at)) == 0)
D = Dict(V)
V1 = JuLIP.decode_dict(D)
println(@test V1 == V)

@info("Testing M-One-Body")
Vm = MOneBody(:Si => E0)
println(@test energy(Vm, at) ≈ energy(V, at))

E1 = rand()
V2 = MOneBody(:Si => E0, :C => E1)
at.Z[2:2:end] .= atomic_number(:C)
println(@test energy(V2, at) ≈ length(at) * (E0+E1) / 2)
println(@test norm(forces(V, at)) == 0)
D = Dict(V2)
V21 = JuLIP.decode_dict(D)
println(@test V2 == V21)
