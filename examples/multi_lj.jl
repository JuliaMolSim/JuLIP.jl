

module MultiLJ

using JuLIP
using JuLIP.Potentials: pairs
using LinearAlgebra: I
import JuLIP: energy, forces, cutoff

export MLJ

struct MLJ{T} <: AbstractCalculator
   Z2idx::Vector{Int}
   V::Matrix{T}
   rcut::Float64
end

@pot MLJ

cutoff(V::MLJ) = V.rcut

function MLJ(Z, ϵ, σ; rcutfact = 2.5)
   @assert length(Z) == length(unique(Z))
   # create a mapping from atomic numbers to indices
   Z2idx = zeros(Int, maximum(Z))
   Z2idx[Z] = 1:length(Z)
   # generate the potential
   rcut = maximum(σ) * rcutfact
   V = [ LennardJones(σ[a,b], ϵ[a, b]) * C2Shift(σ[a,b] * rcutfact)
         for a = 1:length(Z), b = 1:length(Z) ]
   return MLJ(Z2idx, V, rcut)
end

function energy(V::MLJ, at::Atoms)
   E = 0.0
   for (i, j, r, R) in pairs(at, cutoff(V))
      a = V.Z2idx[at.Z[i]]
      b = V.Z2idx[at.Z[j]]
      E += 0.5 * V.V[a, b](r)
   end
   return E
end

function forces(V::MLJ, at::Atoms)
   F = zeros(JVecF, length(at))
   for (i, j, r, R) in pairs(at, cutoff(V))
      a = V.Z2idx[at.Z[i]]
      b = V.Z2idx[at.Z[j]]
      f = 0.5 * grad(V.V[a,b], r, R)
      F[i] += f
      F[j] -= f
   end
   return F
end

end


# --------------
#   test code
# --------------

using JuLIP, MultiLJ, StaticArrays

# parameters from PHYSICAL REVIEW B, VOLUME 64, 184201
#                 Crystals of binary Lennard-Jones solids
#                 T F Middleton, J Hernandez-Rojas, P N Mortenson, D J Wales

ϵ = [1.0 1.5; 1.5 0.5]
σ = [1.0 0.8; 0.8 0.88]
z = [1, 2]
V = MultiLJ.MLJ(z, ϵ, σ)

#
n = 2
x, o = 0:n, ones(n)
X = [ kron(x, o, o)[:]'; kron(o, x, o)'; kron(o, o, x)' ]
X = vecs( X + 0.05 * rand(size(X)) )
Nat = length(X)
at = Atoms(X, zeros(X), ones(Nat), rand(z, Nat), 5 * Matrix(1.0I, 3,3), false;  calc = V)

# check that energy and forces evaluate ok
energy(V, at)
forces(V, at)

# finite-difference test
JuLIP.Testing.fdtest(V, at)
