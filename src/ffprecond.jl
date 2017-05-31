
using JuLIP: vecs, mat
using JuLIP.Potentials: evaluate_d, SitePotential, sites
using ForwardDiff, PositiveFactorizations

import JuLIP.Potentials: cutoff



"""
`FF`: defines a preconditioner based on a force-field;

TODO: thorough documentation and reference once the paper is finished
"""
function FF(at::AbstractAtoms, V::SitePotential;
            tol = 1e-7, updatefreq=10, solver = :amg)
   r0 = estimate_rnn(at)
   if solver == :amg
      return AMGPrecon(V, at, tol=tol, updatedist = 0.2 * r0, updatefreq=updatefreq)
   elseif solver == :direct
      return DirectPrecon(V, at, updatedist=0.2 * r0, tol=tol, updatefreq=updatefreq)
   else
      error("unknown kwarg solver = $(solver)")
   end
end


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
         GJ1, GJ2 = jj[j1], jj[j2]                       # global indices
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
