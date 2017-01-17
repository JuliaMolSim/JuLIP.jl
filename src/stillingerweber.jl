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

import JuLIP.Potentials: evaluate, evaluate_d, @pot, @D

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


@pot type StillingerWeber{P1,P2} <: SitePotential
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
"""
StillingerWeber

cutoff(calc::StillingerWeber) = max(cutoff(calc.V2), cutoff(calc.V3))

StillingerWeber(; ϵ=2.1675, σ = 2.0951, A=7.049556277, B=0.6022245584,
                  p = 4, a = 1.8, λ=21.0, γ=1.20 ) =
   StillingerWeber(
      PairPotential(:( $(0.5 * ϵ * A) * ($B * (r/$σ)^(-$p) - 1.0)
                                 * exp( 1.0 / (r/$σ - $a) ) ),
                        cutoff = a*σ-1e-2),
      PairPotential(:( $(sqrt(ϵ * λ)) * exp( $γ / (r/$σ - $a) ) ),
                        cutoff = a*σ-1e-2)
   )

function evaluate(calc::StillingerWeber, r, R)
   Es = 0.0
   # two-body contributions
   for r1 in r
      Es += calc.V2(r1)
   end
   # three-body contributions
   S = [ R1/r1 for (R1,r1) in zip(R, r) ]
   V3 = [ calc.V3(r1) for r1 in r ]
   for i1 = 1:(length(r)-1), i2 = (i1+1):length(r)
      Es += V3[i1] * V3[i2] * bondangle(S[i1], S[i2])
   end
   return Es
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
