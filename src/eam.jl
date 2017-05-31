
# =================== General Single-Species EAM Potential ====================
# TODO: Alloy potential

@pot type EAM{T1, T2, T3} <: SitePotential
   ϕ::T1    # pair potential
   ρ::T2    # electron density potential
   F::T3    # embedding function
end

cutoff(V::EAM) = max(cutoff(V.ϕ), cutoff(V.ρ))

evaluate(V::EAM, r, R) = V.F( sum(t->V.ρ(t), r) ) + 0.5 * sum(t->V.ϕ(t), r)

# TODO: this creates a lot of unnecessary overhead; probaby better to
#       define vectorised versions of pair potentials
function evaluate_d(V::EAM, r, R)
   ρ̄ = sum(V.ρ(s) for s in r)
   dF = @D V.F(ρ̄)
   #         (0.5 * ϕ'          + F' *  ρ')           * ∇r     (where ∇r = R/r)
   return [ ((0.5 * (@D V.ϕ(s)) + dF * (@D V.ρ(s))) / s) * S  for (s, S) in zip(r, R) ]
end

# TODO: which of the two `evaluate_dd` and `hess` should we be using?
evaluate_dd(V::EAM, r, R) = hess(V, r, R)
hess(V::EAM, r, R) = _hess_(V, r, R, identity, hess)

# ff preconditioner specification for EAM potentials
#   (just replace id with abs and hess with precon in the hessian code)
precon(V::EAM, r, R) = _hess_(V, r, R, abs, precon)


function _hess_(V::EAM, r, R, fabs, fhess)
   # allocate storage
   H = zeros(JMatF, length(r), length(r))
   # precompute some stuff
   ρ̄ = sum( V.ρ(s, S)  for (s, S) in zip(r, R) )
   ∇ρ = [ grad(V.ρ, s, S) for (s, S) in zip(r, R) ]
   dF = fabs(@D V.F(ρ̄))
   ddF = fabs(@DD V.F(ρ̄))
   # assemble
   for i = 1:length(r)
      for j = 1:length(r)
         H[i,j] = ddF * ∇ρ[i] * ∇ρ[j]'
      end
      # TODO: this can be made more performant by combining the two
      #       pair-potential-like terms into one big one
      #       and this would also lead to a better preconditioner 
      H[i,i] += 0.5 * fhess(V.ϕ, r[i], R[i]) + dF * fhess(V.ρ, r[i], R[i])
   end
   return H
end



"""
`type EAM`

Generic Single-species EAM potential, to specify it, one needs to
specify the pair potential ϕ, the electron density ρ and the embedding
function F.

The most convenient constructor is likely via tabulated values,
more below.

# Constructors:
```
EAM(pair::PairPotential, eden::PairPotential, embed)
EAM(fpair::AbstractString, feden::AbstractString, fembed::AbstractString; kwargs...)
```

## Constructing an EAM potential from tabulated values

At the moment only the .plt format is implemented. Files can e.g. be
obtained from
* [NIST](https://www.ctcms.nist.gov/potentials/)

Use the `EAM(fpair, feden, fembed)` constructure. The keyword arguments specify
details of how the tabulated values are fitted; see
`?SplinePairPotential` for more details.

TODO: implement other file formats.
"""
EAM


# implementation of EAM models using data files

function EAM(fpair::AbstractString, feden::AbstractString,
             fembed::AbstractString; kwargs...)
   pair = SplinePairPotential(fpair; kwargs...)
   eden = SplinePairPotential(feden; kwargs...)
   embed = SplinePairPotential(feden; fixcutoff = false, kwargs...)
   return EAM(pair, eden, embed)
end


# ================= Finnis-Sinclair Potential =======================


@pot type FSEmbed end
evaluate(V::FSEmbed, ρ̄) = - sqrt(ρ̄)
evaluate_d(V::FSEmbed, ρ̄) = - 0.5 / sqrt(ρ̄)
evaluate_dd(V::FSEmbed, ρ̄) = 0.25 * ρ̄^(-3/2)

"""
`FinnisSinclair`: constructs an EAM potential with embedding function
-√ρ̄.
"""
FinnisSinclair(pair::PairPotential, eden::PairPotential) =
   EAM(pair, eden, FSEmbed())

function FinnisSinclair(fpair::AbstractString, feden::AbstractString; kwargs...)
   pair = SplinePairPotential(fpair; kwargs...)
   eden = SplinePairPotential(feden; kwargs...)
   return FinnisSinclair(pair, eden)
end
