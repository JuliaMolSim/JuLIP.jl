# included from Potentials.jl
# part of the module JuLIP.Potentials

using JuLIP: JVec, JMat, neighbourlist
using LinearAlgebra: I
using JuLIP.Chemistry: atomic_number
using NeighbourLists

export ZeroPairPotential, PairSitePotential, ZBLPotential,
         LennardJones, lennardjones,
         Morse, morse

## TODO: kill this one?
grad(V::PairPotential, r::Real, R::JVec) = ((@D V(r)) / r) * R


evaluate!(tmp, V::PairPotential, r::Union{Number, JVec}) = V(r)
evaluate_d!(tmp, V::PairPotential, r::Union{Number, JVec}) = @D V(r)
evaluate_dd!(tmp, V::PairPotential, r::Union{Number, JVec}) = @DD V(r)

function evaluate!(tmp, V::PairPotential, R::AbstractVector{JVec{T}}) where {T}
   Es = zero(T)
   for i = 1:length(R)
      Es += T(0.5) * evaluate!(tmp, V, norm(R[i]))
   end
   return Es
end

function evaluate_d!(dEs, tmp, V::PairPotential, R::AbstractVector{JVec{T}}) where {T}
   for i = 1:length(R)
      r = norm(R[i])
      dEs[i] = (T(0.5) * evaluate_d!(tmp, V, r) / r) * R[i]
   end
   return dEs
end

function evaluate_dd!(hEs, tmp, V::PairPotential, R::AbstractVector{<: JVec})
   n = length(R)
   for i = 1:n
      hEs[i,i] = 0.5 * _hess!(tmp, V, norm(R[i]), R[i])
   end
   return hEs
end

function _hess!(tmp, V::PairPotential, r::Number, R::JVec)
   R̂ = R/r
   P = R̂ * R̂'
   dV = evaluate_d!(tmp, V, r) / r
   ddV = evaluate_dd!(tmp, V, r)
   return (ddV - dV) * P + dV * I
end

function precon!(hEs, tmp, V::PairPotential, R::AbstractVector{<: JVec}, innerstab=T(0.0))
   n = length(R)
   for i = 1:n
      hEs[i,i] = precon!(tmp, V, norm(R[i]), R[i], innerstab)
   end
   return hEs
end

# an FF preconditioner for pair potentials
function precon!(tmp, V::PairPotential, r::T, R::JVec{T}, innerstab=T(0.1)
                 ) where {T <: Number}
   r = norm(R)
   dV = evaluate_d!(tmp, V, r)
   ddV = evaluate_dd!(tmp, V, r)
   R̂ = R/r
   return (1-innerstab) * (abs(ddV) * R̂ * R̂' + abs(dV / r) * (I - R̂ * R̂')) +
             innerstab  * (abs(ddV) + abs(dV / r)) * I
end



"""
`LennardJones(σ, e0):` constructs the 6-12 Lennard-Jones potential [wiki](https://en.wikipedia.org/wiki/Lennard-Jones_potential)

   e0 * 4 * ( (σ/r)¹² - (σ/r)⁶ )

Constructor with kw arguments: `LennardJones(; kwargs...)`

* `e0, σ` : standard LJ parameters, default `e0 = 1.0, σ = 1.0`
* `r0` : equilibrium distance - if `r0` is specified then `σ` is ignored
* `a0` : FCC lattice parameter, if `a0` is specified then `σ` is ignored

(`r0, a0` cannot both be specified at the same time)
"""
LennardJones(σ, e0) = (@analytic r -> e0 * 4.0 * ((σ/r)^(12) - (σ/r)^(6)))


function ljparams(; σ=1.0, e0=1.0, r0 = nothing, a0 = nothing)
   if r0 != nothing && a0 != nothing
      error("`LenndardJones`: cannot specify both `r0` and `a0`")
   end
   if a0 != nothing    # r0 = nn-dist = a0/sqrt(2) in FCC
      r0 = a0 / sqrt(2)
   end
   if r0 != nothing    # standard LJ is minimised at r = 2^(1/6)
      σ =  r0 / 2^(1/6)
   end
   return σ, e0
end

LennardJones(; kwargs...) = LennardJones(ljparams(;kwargs...)...)


"""
`lennardjones(; kwargs...)`

simplified constructor for `LennardJones` (note this is type unstable!)

In addition to the `kwargs` of `LennardJones`, this accepts also

* `rcut` : default `:auto` which gives `rcut = (1.9*σ, 2.7*σ)`. Use
`nothing` or `Inf` to specify no cutoff, or specify a tuple or
array with two elements specifying the lower and upper cut-off radii to be
used with `SplineCutoff`.
"""
function lennardjones(; rcut = :auto, kwargs...)
   σ, e0 = ljparams(; kwargs...)
   if (rcut == nothing || rcut == Inf)
      return LennardJones(σ, e0)
   elseif rcut == :auto
      rcut = (1.9*σ, 2.7*σ)
   end
   return SplineCutoff(rcut[1], rcut[2]) * LennardJones(σ, e0)
end

"""
`Morse(A, e0, r0)` or `Morse(;A=4.0, e0=1.0, r0=1.0)`: constructs a
`PairPotential` for
```
   e0 ( exp( -2 A (r/r0 - 1) ) - 2 exp( - A (r/r0 - 1) ) )
```
"""
Morse(A, e0, r0) = @analytic(
   r -> e0 * ( exp(-(2.0*A) * (r/r0 - 1.0)) - 2.0 * exp(-A * (r/r0 - 1.0)) ) )
Morse(;A=4.0, e0=1.0, r0=1.0) = Morse(A, e0, r0)

"""
`morse(A=4.0, e0=1.0, r0=1.0, rcut=(1.9*r0, 2.7*r0))`

simplified constructor for `Morse` (type unstable)
"""
morse(;A=4.0, e0=1.0, r0=1.0, rcut=(1.9*r0, 2.7*r0)) = (
   (rcut == nothing || rcut == Inf)
         ?  Morse(A, e0, r0)
         :  SplineCutoff(rcut[1], rcut[2]) * Morse(A, e0, r0) )


"""
`ZeroPairPotential()`: creates a potential that just returns zero
"""
struct ZeroPairPotential <: PairPotential end

@pot ZeroPairPotential

evaluate(p::ZeroPairPotential, r::T) where {T <: Number} = T(0.0)
evaluate_d(p::ZeroPairPotential, r::T) where {T <: Number} = T(0.0)
evaluate_dd(p::ZeroPairPotential, r::T) where {T <: Number} = T(0.0)
cutoff(p::ZeroPairPotential) = Bool(0) # the weakest number type



# ------------------------------------------------------------------------

# TODO: write more docs + tests for ZBL

"""
Implementation of the ZBL potential to model close approach.
"""
struct ZBLPotential{TV} <: PairPotential
   Z1::Int
   Z2::Int
   V::TV   # analytic
end

@pot ZBLPotential

evaluate(V::ZBLPotential, r::Number) = evaluate(V.V, r::Number)
evaluate_d(V::ZBLPotential, r::Number) = evaluate_d(V.V, r::Number)
cutoff(::ZBLPotential) = Inf

ZBLPotential(Z1::Integer, Z2::Integer) =
   let Z1=Z1, Z2=Z2
      au = 0.8854 * 0.529 / (Z1^0.23 + Z2^0.23)
      ϵ0 = 0.00552634940621
      C = Z1*Z2/(4*π*ϵ0)
      E1, E2, E3, E4 = 0.1818, 0.5099, 0.2802, 0.02817
      A1, A2, A3, A4 = 3.2/au, 0.9423/au, 0.4028/au, 0.2016/au
      V = @analytic(r -> C * (E1*exp(-A1*r) + E2*exp(-A2*r) +
                              E3*exp(-A4*r) + E4*exp(-A4*r) ) / r)
      ZBLPotential(Z1, Z2, V)
   end


ZBLPotential(Z::Integer) = ZBLPotential(Z, Z)
ZBLPotential(s1::Symbol, s2::Symbol) = ZBLPotential(atomic_number(s1), atomic_number(s2))
ZBLPotential(s::Symbol) = ZBLPotential(s, s)

Dict(V::ZBLPotential) = Dict("__id__" => "JuLIP_ZBLPotential",
                             "Z1" => V.Z1,
                             "Z2" => Z.Z2)
ZBLPotential(D::Dict) = ZBLPotential(D["Z1"], D["Z2"])
Base.convert(::Val{:JuLIP_ZBLPotential}, D::Dict) = ZBLPotential(D)


# ====================================================================
#   A product of two pair potentials: primarily used for cutoff mechanisms

"product of two pair potentials"
mutable struct ProdPot{P1, P2} <: PairPotential
   p1::P1
   p2::P2
end

@pot ProdPot

import Base.*
*(p1::PairPotential, p2::PairPotential) = ProdPot(p1, p2)
@inline evaluate(p::ProdPot, r::Number) = p.p1(r) * p.p2(r)
evaluate_d(p::ProdPot, r::Number) = (p.p1(r) * (@D p.p2(r)) + (@D p.p1(r)) * p.p2(r))
evaluate_dd(p::ProdPot, r::Number) = (p.p1(r) * (@DD p.p2(r)) +
              2 * (@D p.p1(r)) * (@D p.p2(r)) + (@DD p.p1(r)) * p.p2(r))
cutoff(p::ProdPot) = min(cutoff(p.p1), cutoff(p.p2))

# ====================================================================



"""
`struct WrappedPairPotential`

wraps a pairpotential using `FunctionWrappers` in order to allow
type-stable storage of multiple potentials. This is the main technique
required at the moment to work with multi-component systems.
Otherwise, this is not advisable since it disables a range of
possible compiler optimisations.
"""
struct WrappedPairPotential <: SimplePairPotential
   f::F64fun
   f_d::F64fun
   f_dd::F64fun
   rcut::Float64
end

@pot WrappedPairPotential

cutoff(V::WrappedPairPotential) = V.rcut
# evaluate, etc are all derived from SimplePairPotential

function WrappedPairPotential(V::PairPotential)
   @assert (0 < cutoff(V) < Inf)
   f, f_d, f_dd = let V=V, rc = cutoff(V)
      (F64fun(r -> evaluate(V, r) * (r<rc)),
              F64fun(r -> evaluate_d(V, r) * (r<rc)),
              F64fun(r -> evaluate_dd(V, r) * (r<rc)))
   end
   return WrappedPairPotential(f, f_d, f_dd, cutoff(V))
end
