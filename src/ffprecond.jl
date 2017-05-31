
using JuLIP: vecs, mat
using JuLIP.Potentials: evaluate_d, SitePotential, sites
using ForwardDiff, PositiveFactorizations

import JuLIP.Potentials: cutoff



# TODO: fine-tune and tweak (see Exp for comparison)
function FF(at::AbstractAtoms, V; tol = 1e-7)
   r0 = estimate_rnn(at)
   return AMGPrecon(V, at, tol=tol)
end

Exp(at::AbstractAtoms;
             A=3.0, r0=estimate_rnn(at), cutoff_mult=2.2,
             tol=1e-7, updatefreq=10, solver = :amg, energyscale = 1.0)




hinds(j) =  3 * (j-1) + [1,2,3]


   # # convert to a matrix
   # P = zeros(3*n, 3*n)
   # for i1 = 1:n, i2 = 1:n
   #    I1 = 3*(i1-1) + [1,2,3]
   #    I2 = 3*(i2-1) + [1,2,3]
   #    P[I1, I2] = pV[i1, i2]
   # end
   # P = 0.5 * (P + P')
   # return P

"""
build the preconditioner matrix associated with a site potential
"""
function matrix(V::SitePotential, at::AbstractAtoms)
   I = Int[]; J = Int[]; Z = Float64[]
   for (i0, neigs, r, R, _) in sites(at, cutoff(V))
      ii = atind2lininds(i0)
      jj = [atind2lininds(j_) for j_ in neigs]

      # compute positive version of hessian of V(R)
      hV = precon(V, r, R)

      nneigs = length(neigs)
      for j1 = 1:nneigs, j2 = 1:nneigs
         # LJ1, LJ2 = hinds(j1), hinds(j2) # local indices
         GJ1, GJ2 = jj[j1], jj[j2]                       # global indices
         # H = view(hV, LJ1, LJ2)
         H = hV[j1, j2]
         for a = 1:3, b = 1:3
            if abs(H[a, b]) > 1e-5
               append!(I, [ GJ1[a],    ii[a],  GJ1[a],  ii[a] ])
               append!(J, [ GJ2[b],   GJ2[b],   ii[b],  ii[b] ])
               append!(Z, [ H[a,b], - H[a,b], -H[a,b], H[a,b] ])
            end
         end
      end
   end
   N = 3*length(at)
   return sparse(I, J, Z, N, N) + 0.001 * speye(N)
end
