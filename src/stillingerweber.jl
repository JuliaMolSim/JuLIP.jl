# NOTE: this is a quick and dirty implementation of the Stillinger-Weber model;
# in the future it will be good to do this more elegantly / more efficiently
# there is huge overhead in this code that can be
#   significantly improved in terms of performance, but for now we just
#   want a correct and readable code

# [Stillinger/Weber, PRB 1985]
# ---------------------------------------------------
# v2 = ϵ f2(r_ij / σ); f2(r) = A (B r^{-p} - r^{-q}) exp( (r-a)^{-1} )
# v3 = ϵ f3(ri/σ, rj/σ, rk/σ); f3 = h(rij, rij, Θjik) + ... + ...
# h(rij, rik, Θjik) = λ exp[ γ (rij-a)^{-1} + γ (rik-a)^{-1} ] * (cosΘjik+1/3)^2
#       >>>
# V2 = 0.5 * ϵ * A (B r^{-p} - r^{-q}) * exp( (r-a)^{-1} )
# V3 = √ϵ * λ exp[ γ (r-a)^{-1} ]
#
# Parameters from QUIP database:
# -------------------------------
# <per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
         # p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
# <per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14" lambda="21.0"
         # gamma="1.20" eps="2.1675" />


using ForwardDiff
using LinearAlgebra: dot

export StillingerWeber



"""
`bondangle(S1, S2) -> (dot(S1, S2) + 1.0/3.0)^2`

* not this assumes that `S1, S2` are normalised
* see `bondangle_d` for the derivative
"""
bondangle(S1, S2) = (dot(S1, S2) + 1.0/3.0)^2

"""
`b := bondangle(S1, S2)` then

`bondangle_d(S1, S2, r1, r2) -> b, db1, db2`

where `dbi` is the derivative of `b` w.r.t. `Ri` where `Si= Ri/ri`.
"""
function bondangle_d(S1, S2, r1, r2)
   d = dot(S1, S2)
   b1 = (1.0/r1) * S2 - (d/r1) * S1
   b2 = (1.0/r2) * S1 - (d/r2) * S2
   return (d+1.0/3.0)^2, 2.0*(d+1.0/3.0)*b1, 2.0*(d+1.0/3.0)*b2
end


function _ad_bondangle_(R)
   R1, R2 = R[1:3], R[4:6]
   r1, r2 = norm(R1), norm(R2)
   return bondangle(R1/r1, R2/r2)
end

# TODO: need a faster implementation of bondangle_dd
function bondangle_dd(R1, R2)
   R = [R1; R2]
   hh = ForwardDiff.hessian(_ad_bondangle_, R)
   h = zeros(JMatF, 2,2)
   h[1,1] = JMatF(hh[1:3,1:3])
   h[1,2] = JMatF(hh[1:3, 4:6])
   h[2,1] = JMatF(hh[4:6,1:3])
   h[2,2] = JMatF(hh[4:6, 4:6])
   return h
end


"""
Stillinger-Weber potential with parameters for Si.

Functional form and default parameters match the original SW potential
from [Stillinger/Weber, PRB 1985].

The `StillingerWeber` type can also by "abused" to generate arbitrary
bond-angle potentials of the form
   ∑_{i,j} V2(rij) + ∑_{i,j,k} V3(rij) V3(rik) (cos Θijk + 1/3)^2

Constructor admits the following key-word parameters:
`ϵ=2.1675, σ = 2.0951, A=7.049556277, B=0.6022245584,
                  p = 4, a = 1.8, λ=21.0, γ=1.20`

which enter the potential as follows:
```
V2(r) = 0.5 * ϵ * A * (B * (r/σ)^(-p) - 1.0) * exp(1.0 / (r/σ - a))
V3(r) = sqrt(ϵ * λ) * exp(γ / (r/σ - a))
```
"""
struct StillingerWeber{P1,P2} <: SitePotential
   V2::P1
   V3::P2
end

@pot StillingerWeber

cutoff(calc::StillingerWeber) = max(cutoff(calc.V2), cutoff(calc.V3))


function StillingerWeber(; brittle = false,
               ϵ=2.1675, σ = 2.0951, A=7.049556277, B=0.6022245584,
               p = 4, a = 1.8, λ = brittle ? 42.0 : 21.0, γ=1.20 )
   cutoff = a*σ-1e-2
   V2 = @analytic(r -> (ϵ*A) * (B*(r/σ)^(-p) - 1.0) * exp(1.0/(r/σ - a))) *
         HS(cutoff)
   V3 = @analytic(r -> sqrt(ϵ * λ) * exp( γ / (r/σ - a) )) * HS(cutoff)
   return StillingerWeber(V2, V3)
end

function evaluate(calc::StillingerWeber, r, R)
   # two-body contributions
   if length(r) == 0
      return 0.0
   end
   Es = 0.5 * sum(calc.V2, r)
   # three-body contributions
   S = [ R1/r1 for (R1,r1) in zip(R, r) ]    # make a temporary array!
   V3 = [ calc.V3(r1) for r1 in r ]
   for i1 = 1:(length(r)-1), i2 = (i1+1):length(r)
      Es += V3[i1] * V3[i2] * bondangle(S[i1], S[i2])
   end
   return Es
end


function evaluate_d(calc::StillingerWeber, r::AbstractVector{T}, R) where {T}
   # two-body terms
   if length(r) == 0
      return JVec{T}[]
   end
   dEs = [ 0.5 * grad(calc.V2, ri, Ri) for (ri, Ri) in zip(r, R) ]
   # three-body terms
   S = [ R1/r1 for (R1,r1) in zip(R, r) ]
   V3 = [calc.V3(s) for s in r]
   gV3 = [ grad(calc.V3, r1, R1) for (r1, R1) in zip(r, R) ]
   for i1 = 1:(length(r)-1), i2 = (i1+1):length(r)
      a, b1, b2 = bondangle_d(S[i1], S[i2], r[i1], r[i2])
      dEs[i1] += (V3[i1] * V3[i2]) * b1 + (V3[i2] * a) * gV3[i1]
      dEs[i2] += (V3[i1] * V3[i2]) * b2 + (V3[i1] * a) * gV3[i2]
   end
   return dEs
end


function _ad_dV(V::StillingerWeber, R_dofs)
   R = vecs( reshape(R_dofs, 3, length(R_dofs) ÷ 3) )   # TODO: skip reshape step
   r = norm.(R)
   dV = evaluate_d(V, r, R)
   return mat(dV)[:]        # TODO: this comes up quite a bit -> turn it into a `vec` or `dofs` call?
end


function _ad_ddV(V::StillingerWeber, r::AbstractVector{T}, R) where {T}
   ddV = ForwardDiff.jacobian( Rdofs -> _ad_dV(V, Rdofs), mat(R)[:] )
   # convert into a block-format
   n = length(r)
   hV = zeros(JMat{T}, n, n)
   for i = 1:n, j = 1:n
      hV[i, j] = ddV[ ((i-1)*3).+(1:3), ((j-1)*3).+(1:3) ]
   end
   return hV
end

hess(V::StillingerWeber, r, R) = _ad_ddV(V, r, R)

# function hess(V::StillingerWeber, r, R)
#    n = length(r)
#    hV = zeros(JMatF, n, n)
#
#    # two-body contributions
#    for (i, (r_i, R_i)) in enumerate(zip(r, R))
#       hV[i,i] += hess(V.V2, r_i, R_i)
#    end
#
#    # three-body terms
#    S = [ R1/r1 for (R1,r1) in zip(R, r) ]
#    V3 = [ V.V3(r1) for r1 in r ]
#    dV3 = [ grad(V.V3, r1, R1) for (r1, R1) in zip(r, R) ]
#    hV3 = [ hess(V.V3, r1, R1) for (r1, R1) in zip(r, R) ]
#
#    for i1 = 1:(length(r)-1), i2 = (i1+1):length(r)
#       # Es += V3[i1] * V3[i2] * bondangle(S[i1], S[i2])
#       # precompute quantities
#       ψ, Dψ_i1, Dψ_i2 = bondangle_d(S[i1], S[i2], r[i1], r[i2])
#       Hψ = bondangle_dd(R[i1], R[i2])  # <<<< this should be SLOW (AD)
#       # assemble local hessian contributions
#       hV[i1,i1] +=
#          hV3[i1] * V3[i2] * ψ       +   dV3[i1] * V3[i2] * Dψ_i1' +
#          Dψ_i1 * V3[i2] * dV3[i1]'  +   V3[i1] * V3[i2] * Hψ[1,1]
#       hV[i2,i2] +=
#          V3[i2] * hV3[i2] * ψ       +   V3[i1] * dV3[i2] * Dψ_i2' +
#          Dψ_i2 * V3[i1] * dV3[i2]'  +   V3[i1] * V3[i2] * Hψ[2,2]
#       hV[i1,i2] +=
#          dV3[i1] * dV3[i2]' * ψ     +   V3[i1] * Dψ_i1 * dV3[i2]' +
#          dV3[i1] * V3[i2] * Dψ_i2'  +   V3[i1] * V3[i2] * Hψ[1,2]
#       hV[i2, i1] +=
#          dV3[i2] * dV3[i1]' * ψ     +   V3[i1] * Dψ_i2 * dV3[i1]' +
#          dV3[i2] * V3[i1] * Dψ_i1'  +   V3[i1] * V3[i2] * Hψ[2,1]
#    end
#    return hV
# end



function precon(V::StillingerWeber, r::AbstractVector{T}, R, innerstab=0.0
               ) where {T}
   n = length(r)
   pV = zeros(JMat{T}, n, n)

   # two-body contributions
   for (i, (r1, R1)) in enumerate(zip(r, R))
      pV[i,i] += precon(V.V2, r1, R1)
   end

   # three-body terms
   S = [ R1/r1 for (R1,r1) in zip(R, r) ]
   V3 = V.V3.(r)
   for i1 = 1:(n-1), i2 = (i1+1):n
      Θ = dot(S[i1], S[i2])
      dΘ1 = (T(1.0)/r[i1]) * S[i2] - (Θ/r[i1]) * S[i1]
      dΘ2 = (T(1.0)/r[i2]) * S[i1] - (Θ/r[i2]) * S[i2]
      # ψ = (Θ + 1/3)^2, ψ' = (Θ + 1/3), ψ'' = 2.0
      a = abs((V3[i1] * V3[i2] * T(2.0)))
      pV[i1,i2] += a * dΘ1 * dΘ2'
      pV[i1,i1] += a * dΘ1 * dΘ1'
      pV[i2, i2] += a * dΘ2 * dΘ2'
      pV[i2, i1] += a * dΘ2 * dΘ1'
   end

   return pV
end
