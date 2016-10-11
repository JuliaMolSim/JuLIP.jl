
# this is a quick and dirty implementation of the Stillinger-Weber model;
# in the future it will be good to do this more elegantly.

# Model and model paramters taken from
# https://quantumwise.com/documents/manuals/ATK-2014/ReferenceManual/index.html/ref.stiwe3potential.html
# which seem to be in the right units; should probably get a better reference
# and make sure the translation of units is indeed correct

# NOTE: there is huge overhead in this code that can be
#   significantly improved in terms of performance, but for now we just
#   want a correct and readable code


export StillingerWeber

import JuLIP.Potentials: evaluate, evaluate_d, @pot, @D


bondangle(R1, R2) = 0.5 * ( dot(R1/norm(R1), R2/norm(R2)) + 1.0/3.0 )^2

function bondangle_d(R1, R2)
   d = dot(R1/norm(R1), R2/norm(R2))
   b1 = R2 / (norm(R2)*norm(R1)) - (d/norm(R1)^2) * R1
   b2 = R1 / (norm(R1)*norm(R2)) - (d/norm(R2)^2) * R2
   return (d+1./3.)*b1, (d+1./3.)*b2
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

StillingerWeber(; p=4.0, A = 15.2855528754, B = 0.60222456,
                  cutoff = 3.77118, γ2 = 2.0951, γ3 = 2.51412, λ = 45.5343 ) =
   StillingerWeber(
      AnalyticPotential(:( $A * ($B * r^(-$p) - 1.0) )) * SWCutoff(Rc=cutoff, Lc=γ2),
      SWCutoff(Rc=cutoff, Lc=γ3, e0=2.0*sqrt(λ))
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
      a = bondangle(R[i1], R[i2])
      b1, b2 = bondangle_d(R[i1], R[i2])
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
