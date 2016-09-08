
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

bondangle(R1, R2) = 0.5 * ( dot(R1/norm(R1), R2/norm(R2)) + 1.0/3.0 )^2

function bondangle_d(R1, R2)
   d = dot(R1/norm(R1), R2/norm(R2))
   b1 = R2 / (norm(R2)*norm(R1)) - (d/norm(R1)^2) * R1
   b2 = R1 / (norm(R1)*norm(R2)) - (d/norm(R2)^2) * R2
   return (d+1./3.)*b1, (d+1./3.)*b2
end


"""
Stillinger-Weber potential with parameters for Si.
TODO: add documentation
"""
type StillingerWeber{P1,P2} <: Potential
   V2::P1
   V3::P2
end

cutoff(calc::StillingerWeber) = max(cutoff(calc.V2), cutoff(calc.V3))

StillingerWeber(; p=4.0, A = 15.2855528754, B = 0.60222456,
                  cutoff = 3.77118, γ2 = 2.0951, γ3 = 2.51412, λ = 45.5343 ) =
   StillingerWeber(
      AnalyticPotential(:( $A * ($B * r^(-$p) - 1.0) )) * SWCutoff(Rc=cutoff, Lc=γ2),
      SWCutoff(Rc=cutoff, Lc=γ3, e0=2.0*sqrt(λ))
   )


function site_energies(calc::StillingerWeber, at::AbstractAtoms)
   E = zeros(length(at))
   for (i, neigs, r, R, _) in sites(at, cutoff(calc))
      # two-body contributions
      for r1 in r
         E[i] += calc.V2(r1)
      end
      # three-body contributions
      for i1 = 1:(length(neigs)-1), i2 = (i1+1):length(neigs)
         E[i] += calc.V3(r[i1]) * calc.V3(r[i2]) * bondangle(R[i1], R[i2])
      end
   end
   return E
end

energy(calc::StillingerWeber, at::AbstractAtoms) = sum_kbn(site_energies(calc, at))

function forces(calc::StillingerWeber, at::AbstractAtoms)
   F = zerovecs(length(at))
   for (i, neigs, r, R, _) in sites(at, cutoff(calc))
      # two-body
      for (i1, r1, R1) in zip(neigs, r, R)
         dV2 = ( evaluate_d(calc.V2, r1) / r1 ) * R1
         F[i1] -= dV2
         F[i] += dV2
      end
      # three-body
      for i1 = 1:(length(neigs)-1), i2 = (i1+1):length(neigs)
         a = bondangle(R[i1], R[i2])
         b1, b2 = bondangle_d(R[i1], R[i2])
         V1 = calc.V3(r[i1])
         V2 = calc.V3(r[i2])
         dV1 = (@D calc.V3(r[i1])) * R[i1] / r[i1]
         dV2 = (@D calc.V3(r[i2])) * R[i2] / r[i2]
         F[neigs[i1]] -= V1 * V2 * b1 + dV1 * V2 * a
         F[neigs[i2]] -= V1 * V2 * b2 + V1 * dV2 * a
         F[i] += V1 * V2 * (b1 + b2) + (dV1 * V2 + V1 * dV2) * a
      end
   end
   return F
end
