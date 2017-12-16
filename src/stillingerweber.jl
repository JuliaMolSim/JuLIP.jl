# NOTE: this is a quick and dirty implementation of the Stillinger-Weber model;
# in the future it will be good to do this more elegantly / more efficiently
# there is huge overhead in this code that can be
#   significantly improved in terms of performance, but for now we just
#   want a correct and readable code

# [Stillinger/Weber, PRB 1985]
# ---------------------------------------------------
# v2 = ϵ f2(r_ij / σ); f2(r) = A (B r^{-p} - r^{-q}) exp( (r-a)^{-1} )
# v3 = ϵ f3(ri/σ, rj/σ, rk/σ); f3 = h(rij, rij, Θjik) + ... + ...
# h(rij, rik, Θjik) = λ exp[ γ (rij-a)^{-1} + γ (rik-a)^{-1} ] * (cos Θjik + 1/3)^2
#       >>>
# V2 = 0.5 * ϵ * A (B r^{-p} - r^{-q}) * exp( (r-a)^{-1} )
# V3 = √ϵ * λ exp[ γ (r-a)^{-1} ]
#
# Parameters from QUIP database:
# -------------------------------
# <per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584" p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
# <per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14" lambda="21.0" gamma="1.20" eps="2.1675" />



export StillingerWeber

# using JuLIP.Potentials: evaluate_dd, @D, @DD
# import JuLIP.Potentials: evaluate, evaluate_d


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
   return (d+1./3.)^2, 2.0*(d+1./3.)*b1, 2.0*(d+1./3.)*b2
end


@pot struct StillingerWeber{P1,P2} <: SitePotential
   V2::P1
   V3::P2
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
StillingerWeber

cutoff(calc::StillingerWeber) = max(cutoff(calc.V2), cutoff(calc.V3))

# TODO: brittle StillingerWeber
#       make λ = 42.0
# TODO: implement cutoff for the PairPotential macro
function StillingerWeber(; brittle = false,
               ϵ=2.1675, σ = 2.0951, A=7.049556277, B=0.6022245584,
               p = 4, a = 1.8, λ = brittle ? 42.0 : 21.0, γ=1.20 )
   cutoff = a*σ-1e-2
   V2 = @analytic(r -> (0.5*ϵ*A) * (B*(r/σ)^(-p) - 1.0) * exp(1.0/(r/σ - a))) * HS(cutoff)
   V3 = @analytic(r -> sqrt(ϵ * λ) * exp( γ / (r/σ - a) )) * HS(cutoff)
   return StillingerWeber(V2, V3)
end

function evaluate(calc::StillingerWeber, r, R)
   # two-body contributions
   Es = sum(calc.V2, r)

   # three-body contributions
   S = [ R1/r1 for (R1,r1) in zip(R, r) ]
   V3 = [ calc.V3(r1) for r1 in r ]
   for i1 = 1:(length(r)-1), i2 = (i1+1):length(r)
      Es += V3[i1] * V3[i2] * bondangle(S[i1], S[i2])
   end
   return Es
end

function energy(calc::StillingerWeber, at::ASEAtoms)
   nlist = neighbourlist(at, cutoff(calc))

   # 2-body contribution
   E = 2 * sum(calc.V2, nlist.r)

   # 3-body contribution
   V3 = [calc.V3(r)  for r in nlist.r]
   n = 0
   for idx = 1:length(at)
      n += 1
      a = n
      while n < length(nlist) && nlist.i[n+1] == idx
         n += 1
      end
      b = n
      for i1 = a:(b-1), i2 = (i1+1):(b)
         E += V3[i1] * V3[i2] * bondangle(nlist.R[i1]/nlist.r[i1], nlist.R[i2]/nlist.r[i2])
      end
   end

   return E
end


function evaluate_d(calc::StillingerWeber, r, R)
   # two-body terms
   dEs = [ grad(calc.V2, ri, Ri) for (ri, Ri) in zip(r, R) ]
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


function forces(calc::StillingerWeber, at::ASEAtoms)
   nlist = neighbourlist(at, cutoff(calc))

   # pair potential contribution to forces
   dE = zerovecs(length(at))
   for n = 1:length(nlist)
      dE[nlist.i[n]] += 2 * grad(calc.V2, nlist.r[n], nlist.R[n])
   end

   # 3-body contribution
   V3 = [calc.V3(r)  for r in nlist.r]
   dV3 = [(@D calc.V3(r))/r  for r in nlist.r]
   n = 0
   for idx = 1:length(at)
      n += 1
      a = n
      while n < length(nlist) && nlist.i[n+1] == idx
         n += 1
      end
      b = n
      for i1 = a:(b-1), i2 = (i1+1):(b)
         α, b1, b2 = bondangle_d(nlist.R[i1]/nlist.r[i1],
                           nlist.R[i2]/nlist.r[i2], nlist.r[i1], nlist.r[i2])
         f1 = (V3[i1] * V3[i2]) * b1 + ((V3[i2] * α) * dV3[i1]) * nlist.R[i1]
         f2 = (V3[i1] * V3[i2]) * b2 + ((V3[i1] * α) * dV3[i2]) * nlist.R[i2]
         dE[nlist.j[i1]] -= f1
         dE[nlist.i[i1]] += f1
         dE[nlist.j[i2]] -= f2
         dE[nlist.i[i2]] += f2
      end
   end
   return dE
end


function precon(V::StillingerWeber, r, R)
   n = length(r)
   pV = zeros(JMatF, n, n)

   # two-body contributions
   for (i, (r1, R1)) in enumerate(zip(r, R))
      pV[i,i] += precon(V.V2, r1, R1)
   end

   # three-body terms
   S = [ R1/r1 for (R1,r1) in zip(R, r) ]
   V3 = V.V3.(r)
   # gV3 = [ grad(calc.V3, r1, R1) for (r1, R1) in zip(r, R) ]
   # pV3 = [ precon(calc.V3, r1, R1) for (r1, R1) in zip(r, R) ]
   for i1 = 1:(n-1), i2 = (i1+1):n
      Θ = dot(S[i1], S[i2])
      dΘ1 = (1.0/r[i1]) * S[i2] - (Θ/r[i1]) * S[i1]
      dΘ2 = (1.0/r[i2]) * S[i1] - (Θ/r[i2]) * S[i2]
      # ψ = (Θ + 1/3)^2, ψ' = (Θ + 1/3), ψ'' = 2.0
      a = abs((V3[i1] * V3[i2] * 2.0))
      pV[i1,i2] += a * dΘ1 * dΘ2'
      pV[i1,i1] += a * dΘ1 * dΘ1'
      pV[i2, i2] += a * dΘ2 * dΘ2'
      pV[i2, i1] += a * dΘ2 * dΘ1'
   end

   return pV
end
