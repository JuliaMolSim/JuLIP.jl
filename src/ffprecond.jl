
using JuLIP: vecs, mat
using JuLIP.Potentials: evaluate_d, SitePotential, sites
using ForwardDiff, PositiveFactorizations

import JuLIP.Potentials: cutoff

type ADP{VT <: SitePotential} <: SitePotential
   V::VT
end
cutoff(V::ADP) = cutoff(V.V)

# TODO: fine-tune and tweak (see Exp for comparison)
function FF(at::AbstractAtoms, V; tol = 1e-7)
   r0 = estimate_rnn(at)
   return AMGPrecon(V, at, tol=tol)   # , updatedist = 0.2 * r0
end

function _ad_grad(V::SitePotential, S)
   R = reshape(S, 3, length(S) ÷ 3) |> vecs
   r = [norm(u) for u in R]
   dV = evaluate_d(V, r, R)
   return mat(dV)[:]
end

_ad_hess(V::SitePotential, S) = ForwardDiff.jacobian( S_ -> _ad_grad(V, S_), S )

function precon(Vad::ADP, r, R)
   V = Vad.V
   hV = _ad_hess(V, mat(R)[:])
   # positive factorisation
   hV = 0.5 * (hV + hV')
   L = cholfact(Positive, hV)[:L]
   H = L * L'
   return 0.9 * H + 0.1 * maximum(diag(H)) * eye(size(L,1))
end

hinds(j) =  3 * (j-1) + [1,2,3]


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
         LJ1, LJ2 = hinds(j1), hinds(j2) # local indices
         GJ1, GJ2 = jj[j1], jj[j2]                       # global indices
         H = view(hV, LJ1, LJ2)
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
