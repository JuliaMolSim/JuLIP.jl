import ForwardDiff

import ReverseDiffPrototype
const _RD_ = ReverseDiffPrototype

using JuLIP: vecs

export FDPotential, FDPotential_r, RDPotential_r


# Implementation of a ForwardDiff site potential
# ================================================

function ad_evaluate end

"""
`abstract FDPotential <: SitePotential`

To implement a site potential that uses ForwardDiff.jl for AD, create
a concrete type and overload `JuLIP.Potentials.ad_evaluate`.

Example
```julia
@pot type P1 <: FDPotential
end
JuLIP.Potentials.ad_evaluate{T<:Real}(pot::P1, R::Matrix{T}) =
               sum( exp(-norm(R[:,i])) for i = 1:size(R,2) )
```

Notes:

* For an `FDPotential`, the argument to `ad_evaluate` will be
`R::Matrix` rather than `R::JVecsF`; this is an unfortunate current limitation of
`ForwardDiff`. Hopefully it can be fixed.
* TODO: allow arguments `(r, R)` then use chain-rule to combine them.
"""
abstract FDPotential <: SitePotential

evaluate(pot::FDPotential, r, R) = ad_evaluate(pot, mat(collect(R)))

evaluate_d(pot::FDPotential, r, R) =
   ForwardDiff.gradient( R_ -> ad_evaluate(pot, R_), mat(collect(R)) ) |> vecs


abstract FDPotential_r <: SitePotential

evaluate(pot::FDPotential_r, r, R) = ad_evaluate(pot, collect(r))

function evaluate_d(pot::FDPotential_r, r, R)
   d = ForwardDiff.gradient( r_ -> ad_evaluate(pot, r_), collect(r) )
   return [ (d[i]/r[i]) * R[i] for i = 1:length(r) ]
end


# Implementation of a ReverseDiffPrototype site potential
# ========================================================

abstract RDPotential_r <: SitePotential

evaluate(pot::RDPotential_r, r, R) = ad_evaluate(pot, collect(r))

function evaluate_d(pot::RDPotential_r, r, R)
   d = _RD_.gradient( r_ -> ad_evaluate(pot, r_), collect(r) )
   return [ (d[i]/r[i]) * R[i] for i = 1:length(r) ]
end


# # Implementation of a ReverseDiffSource site potential
# # ========================================================
#
# abstract RDSPotential_r <: SitePotential
#
# init(pot::RDSPotential_r,
#
# evaluate(pot::RDPotential_r, r, R) = ad_evaluate(pot, collect(r))
#
# function evaluate_d(pot::RDPotential_r, r, R)
#    d = _RD_.gradient( r_ -> ad_evaluate(pot, r_), collect(r) )
#    return [ (d[i]/r[i]) * R[i] for i = 1:length(r) ]
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
