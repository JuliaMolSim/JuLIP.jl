# included from Potentials.jl
# part of the module JuLIP.Potentials

# import JuLIP: r_sum

# a simplified way to calculate gradients of pair potentials
grad(p::PairPotential, r::Float64, R::JVec) =
            (evaluate_d(p::PairPotential, r) / r) * R


"`PairCalculator` : basic calculator for pair potentials."
type PairCalculator <: AbstractCalculator
    pp::PairPotential
end

cutoff(calc::PairCalculator) = cutoff(calc.pp)

function energy(calc::PairCalculator, at::ASEAtoms)
   E = 0.0
   for (_,_,r,_,_) in bonds(at)
      E += calc.pp(r)
   end
   return E
end

function forces(calc::PairCalculator, at::ASEAtoms)
   dE = zerovecs(length(at))
   for (i,j,r,R,_) in bonds(at)
      gradV = @GRAD calc.pp(r, R)   # ∇ϕ(|R|) = (ϕ'(r)/r) R
      dE[j] += gradV
      dE[i] -= gradV
   end
   return dE
end

# ------------------------------------------------------------------


"`ZeroPairPotential`: pair potential V(r) = 0.0"
type ZeroPairPotential <: PairPotential end
evaluate(p::ZeroPairPotential, r) = zeros(size(r))
evaluate_d(p::ZeroPairPotential, r) = zeros(size(r))
cutoff(p::ZeroPairPotential) = 0.0

# ------------------------------------------------------------------

"""
`LennardJonesPotential:` e0 * ( (r0/r)¹² - 2 (r0/r)⁶ )

Constructor: `LennardJonesPotential(;r0=1.0, e0=1.0)`
"""
type LennardJonesPotential <: PairPotential
    r0::Float64
    e0::Float64
end
LennardJonesPotential(; r0=1.0, e0=1.0) = LennardJonesPotential(r0, e0)
evaluate(p::LennardJonesPotential, r) = p.e0 * ((p.r0./r).^12 - 2.0 * (p.r0./r).^6)
evaluate_d(p::LennardJonesPotential, r) = -12.0 * p.e0 * ((p.r0./r).^12 - (p.r0./r).^6) ./ r


# ------------------------------------------------------------------


"""
`MorsePotential`: e0 ( exp( -2 A (r/r0 - 1) ) - 2 exp( - A (r/r0 - 1) ) )

Constructor: `MorePotential(;A=4.0, e0=1.0, r0=1.0)`
"""
type MorsePotential <: PairPotential
    e0::Float64
    A::Float64
    r0::Float64
end
MorsePotential(;A=4.0, e0=1.0, r0=1.0) = MorsePotential(e0, A, r0)
@inline morse_exp(p::MorsePotential, r) = exp(-p.A * (r/p.r0 - 1.0))
@inline function evaluate(p::MorsePotential, r)
    e = morse_exp(p, r); return p.e0 * e .* (e - 2.0) end
@inline function  evaluate_d(p::MorsePotential, r)
    e = morse_exp(p, r);  return (-2.0 * p.e0 * p.A / p.r0) * e .* (e - 1.0) end

# ------------------------------------------------------------------

"""
`SimpleExponential`: A exp( B (r/r0 - 1) )

Auxiliary for other potentials, e.g., Gupta.
"""
type SimpleExponential <: PairPotential
  A::Float64
  B::Float64
  r0::Float64
end
evaluate(p::SimpleExponential, r) = p.A * exp( p.B * (r/p.r0 - 1.0) )
evaluate_d(p::SimpleExponential, r) = p.A*p.B/p.r0 * exp( p.B * (r/p.r0 - 1.0) )

# ------------------------------------------------------------------
