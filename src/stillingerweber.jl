
# NOTE: this is a quick and dirty implementation of the Stillinger-Weber model;
# in the future it will be good to do this more elegantly / more efficiently
# there is huge overhead in this code that can be
#   significantly improved in terms of performance, but for now we just
#   want a correct and readable code

# Functional Form from [Stillinger/Weber, PRB 1985]
# ---------------------------------------------------
# v2 = ϵ f2(r_ij / σ)
# v3 = ϵ f3(ri/σ, rj/σ, rk/σ)
# f2(r) = A (B r^{-p} - r^{-q}) exp( (r-a)^{-1} )
# f3 = h(rij, rij, Θjik) + ... + ...
# h(rij, rik, Θjik) = λ exp[ γ (rij-a)^{-1} + γ (rik-a)^{-1} ] * (cos Θjik + 1/3)^2
#
# V2 = 0.5 * ϵ * A (B r^{-p} - r^{-q}) * exp( (r-a)^{-1} )
# V3 = √ϵ * λ exp[ γ (r-a)^{-1} ]
#
#
# Parameters from QUIP database:
# -------------------------------
# <per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
#           p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
# <per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14"
#               lambda="21.0" gamma="1.20" eps="2.1675" />



export StillingerWeber

import JuLIP.Potentials: evaluate, evaluate_d, @pot, @D


iprhat(R1, R2) = dot(R1/norm(R1), R2/norm(R2))

bondangle(R1, R2) = 0.5 * (iprhat(R1, R2) + 1.0/3.0)^2

function bondangle_d(R1, R2)
   d = dot(R1/norm(R1), R2/norm(R2))
   b1 = R2 / (norm(R2)*norm(R1)) - (d/norm(R1)^2) * R1
   b2 = R1 / (norm(R1)*norm(R2)) - (d/norm(R2)^2) * R2
   return 0.5*(d+1./3.)^2, (d+1./3.)*b1, (d+1./3.)*b2
end


@pot type StillingerWeber{P1,P2} <: SitePotential
   V2::P1
   V3::P2
end

"""
Stillinger-Weber potential with parameters for Si.
TODO: add documentation
"""
StillingerWeber

cutoff(calc::StillingerWeber) = max(cutoff(calc.V2), cutoff(calc.V3))

StillingerWeber(; ϵ=2.1675, σ = 2.0951, A=7.049556277, B=0.6022245584,
                  p = 4, a = 1.8, λ=21.0, γ=1.20 ) =
   StillingerWeber(
      AnalyticPotential(:( $(0.5 * ϵ * A) * ($B * (r/$σ)^(-$p) - 1.0)
                                 * exp( 1.0 / (r/$σ - $a) ) ),
                        cutoff = a*σ-1e-3),
      AnalyticPotential(:( $(sqrt(ϵ) * λ) * exp( $γ / (r/$σ - $a) ) ),
                        cutoff = a*σ-1e-3)
   )

function evaluate(calc::StillingerWeber, r, R)
   Es = 0.0
   # two-body contributions
   for r1 in r
      Es += calc.V2(r1)
   end
   # three-body contributions
   for i1 = 1:(length(r)-1), i2 = (i1+1):length(r)
      Es += calc.V3(r[i1]) * calc.V3(r[i2]) * bondangle(R[i1], R[i2])
   end
   return Es
end

function evaluate_d(calc::StillingerWeber, r, R)
   # two-body terms
   # TODO: why can't I use @D here??????
   dEs = [ (evaluate_d(calc.V2, ri) / ri) * Ri for (ri, Ri) in zip(r, R) ]
   # three-body terms
   for i1 = 1:(length(r)-1), i2 = (i1+1):length(r)
      a, b1, b2 = bondangle_d(R[i1], R[i2])
      V1 = calc.V3(r[i1])
      V2 = calc.V3(r[i2])
      dV1 = (@D calc.V3(r[i1])) * R[i1] / r[i1]
      dV2 = (@D calc.V3(r[i2])) * R[i2] / r[i2]
      dEs[i1] += V1 * V2 * b1 + dV1 * V2 * a
      dEs[i2] += V1 * V2 * b2 + V1 * dV2 * a
   end
   return dEs
end

site_energies(calc::StillingerWeber, at::AbstractAtoms) =
   [ calc(r, R) for (_₁, _₂, r, R, _₃) in sites(at, cutoff(calc)) ]

energy(calc::StillingerWeber, at::AbstractAtoms) = sum_kbn(site_energies(calc, at))
