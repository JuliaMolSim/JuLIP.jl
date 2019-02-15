

import ForwardDiff

using JuLIP: vecs, mat

export ADPotential


# Implementation of a ForwardDiff site potential
# ================================================

"""
`abstract FDPotential <: SitePotential`

To implement a site potential that uses ForwardDiff.jl for AD, create
a concrete type and overload `JuLIP.Potentials.ad_evaluate`.

Example
```julia
mutable struct P1 <: FDPotential
end
@pot P1
JuLIP.Potentials.ad_evaluate(pot::P1, R::Matrix{T}) where {T<:Real} =
               sum( exp(-norm(R[:,i])) for i = 1:size(R,2) )
```

Notes:

* For an `FDPotential`, the argument to `ad_evaluate` will be
`R::Matrix` rather than `R::JVecsF`; this is an unfortunate current limitation of
`ForwardDiff`. Hopefully it can be fixed.
* TODO: allow arguments `(r, R)` then use chain-rule to combine them.
"""
struct ADPotential{TV, T, FR} <: SitePotential
   V::TV
   rcut::T
   gradfun::FR
end

@pot ADPotential

ADPotential(V, rcut) = ADPotential(V, rcut, ForwardDiff.gradient)

cutoff(V::ADPotential) = V.rcut

evaluate(V::ADPotential, r, R) = V.V(R)

# evaluate_d(V::ADPotential, r, R) =
#    ForwardDiff.gradient( S -> V.V(vecs(S)), mat(R)[:] ) |> vecs

evaluate_d(V::ADPotential, r, R) =
   V.gradfun( S -> V.V(vecs(S)), collect(mat(R)[:]) ) |> vecs |> collect

# function evaluate_dd(V::ADPotential, r, R)
#
# end



####### Prototype AD Hessian Implementation
# using JuLIP, JuLIP.Potentials
# using JuLIP.Potentials: evaluate_d
# JJ = JuLIP
# JPP = JuLIP.Preconditioners
#
# function ad_grad(V::SitePotential, S)
#    R = reshape(S, 3, length(S) ÷ 3) |> vecs
#    r = [norm(u) for u in R]
#    dV = evaluate_d(V, r, R)
#    return mat(dV)[:]
# end
#
# ad_hess(V::SitePotential, S::Vector{Float64}) = ForwardDiff.jacobian( S_ -> ad_grad(V, S_), S )
#
# ad_hess(V, R::AbstractVector{JVecF}) = ad_hess(V, mat(R)[:])
#
# atind2lininds(i::Integer) = (i-1) * 3 + [1,2,3]
#
# localinds(j::Integer) =  3 * (j-1) + [1,2,3]
#
# function ad_hessian(V, at::AbstractAtoms)
#    I = Int[]; J = Int[]; Z = Float64[]
#    for (i0, neigs, r, R, _) in sites(at, cutoff(V))
#       ii = atind2lininds(i0)
#       jj = [atind2lininds(j_) for j_ in neigs]
#       # compute positive version of hessian of V(R)
#       hV = ad_hess(V, R)
#
#       nneigs = length(neigs)
#       for j1 = 1:nneigs, j2 = 1:nneigs
#          LJ1, LJ2 = localinds(j1), localinds(j2)       # local indices
#          GJ1, GJ2 = jj[j1], jj[j2]                     # global indices
#          H = view(hV, LJ1, LJ2)
#          for a = 1:3, b = 1:3
#             append!(I, [ GJ1[a],    ii[a],  GJ1[a],  ii[a] ])
#             append!(J, [ GJ2[b],   GJ2[b],   ii[b],  ii[b] ])
#             append!(Z, [ H[a,b], - H[a,b], -H[a,b], H[a,b] ])
#          end
#       end
#    end
#    N = 3*length(at)
#    return sparse(I, J, Z, N, N)
# end
#
# ad_hessian(at::AbstractAtoms, x::Vector) =
#     ad_hessian(calculator(at), set_dofs!(at, x))
#



# # ========= STUFF FOR FF Preconditioner =============
#
# """
# `ADP`: this computes a basic FF preconditioner by AD-ing the site
# energy derivatives. It is slow and most of the time not as effective as
# the hand-coded ones.
# """
# type ADP{VT <: SitePotential} <: SitePotential
#  V::VT
# end
# cutoff(V::ADP) = cutoff(V.V)
#
# function _ad_grad(V::SitePotential, S)
#    R = reshape(S, 3, length(S) ÷ 3) |> vecs
#    r = [norm(u) for u in R]
#    dV = evaluate_d(V, r, R)
#    return mat(dV)[:]
# end
#
# _ad_hess(V::SitePotential, S) = ForwardDiff.jacobian( S_ -> _ad_grad(V, S_), S )
#
# function precon(Vad::ADP, r, R)
#    V = Vad.V
#    hV = _ad_hess(V, mat(R)[:])
#    # positive factorisation
#    hV = 0.5 * (hV + hV')
#    L = cholfact(Positive, hV)[:L]
#    H = L * L'
#    return 0.9 * H + 0.1 * maximum(diag(H)) * eye(size(L,1))
# end
#
# hinds(j) =  3 * (j-1) + [1,2,3]
