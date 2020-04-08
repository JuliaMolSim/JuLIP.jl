# included from Potentials.jl
# part of the module JuLIP.Potentials

using JuLIP: JVec, JMat, neighbourlist
using LinearAlgebra: I
using JuLIP.Chemistry: atomic_number
using NeighbourLists

export ZeroPairPotential, ZBLPotential,
         LennardJones, lennardjones,
         Morse, morse

# For PairPotentials we do something kind of weird - we create an intentional
# stack overflow by redirecting
#    evaluate -> evaluate! -> evaluate -> evaluate! -> ...
# A concrete instance of PairPotential must overload either evaluate or
# evaluate! to break this infinite loop. The advantage though is that either
# can be implemented.

evaluate(V::PairPotential, r::Number, z1, z0) =
      evaluate!(alloc_temp(V, 1), V, r, z1, z0)
evaluate_d(V::PairPotential, r::Number, z1, z0) =
      evaluate_d!(alloc_temp_d(V, 1), V, r, z1, z0)
evaluate_dd(V::PairPotential, r::Number, z1, z0) =
      evaluate_dd!(alloc_temp_dd(V, 1), V, r, z1, z0)
evaluate!(tmp, V::PairPotential, r::Number, z1, z0) =
      evaluate(V, r, z1, z0)
evaluate_d!(tmp, V::PairPotential, r::Number, z1, z0) =
      evaluate_d(V, r, z1, z0)
evaluate_dd!(tmp, V::PairPotential, r::Number, z1, z0) =
      evaluate_dd(V, r, z1, z0)

# evaluate!(tmp, V::PairPotential, r::Union{Number, JVec}) = V(r)
# evaluate_d!(tmp, V::PairPotential, r::Number) = @D V(r)
# evaluate_dd!(tmp, V::PairPotential, r::Number) = @DD V(r)
# evaluate_d!(tmp, V::PairPotential, R::JVec) =
#       evaluate_d!(tmp, V, norm(R), R)
# evaluate_d!(tmp, V::PairPotential, r::Number, R::JVec) = ((@D V(r))/r) * R
# evaluate_dd!(tmp, V::PairPotential, r::Number) = @DD V(r)
# evaluate_dd!(tmp, V::PairPotential, R::JVec) =
#       evaluate_dd!(tmp, V, norm(R), R)
# evaluate_dd!(tmp, V::PairPotential, r::Number, R::JVec) =
#       _hess!(tmp, V, r, R)



function evaluate!(tmp, V::PairPotential,
                   R::AbstractVector{JVec{T}}, Z, z0) where {T}
   Es = zero(T)
   for i = 1:length(R)
      Es += T(0.5) * evaluate!(tmp, V, norm(R[i]), Z[i], z0)
   end
   return Es
end

function evaluate_d!(dEs, tmp, V::PairPotential,
                     R::AbstractVector{JVec{T}}, Z, z0) where {T}
   for i = 1:length(R)
      r = norm(R[i])
      dEs[i] = (T(0.5) * evaluate_d!(tmp, V, r, Z[i], z0) / r) * R[i]
   end
   return dEs
end

function evaluate_dd!(hEs, tmp, V::PairPotential,
                      R::AbstractVector{<: JVec}, Z, z0)
   n = length(R)
   for i = 1:n
      hEs[i,i] = 0.5 * _hess!(tmp, V, norm(R[i]), R[i], Z[i], z0)
   end
   return hEs
end

function _hess!(tmp, V::PairPotential, r::Number, R::JVec, z1, z0)
   R̂ = R/r
   P = R̂ * R̂'
   dV = evaluate_d!(tmp, V, r, z1, z0) / r
   ddV = evaluate_dd!(tmp, V, r, z1, z0)
   return (ddV - dV) * P + dV * I
end

function precon!(hEs, tmp, V::PairPotential,
                 R::AbstractVector{<: JVec{T}}, Z, z0,
                 innerstab=T(0.0)) where {T}
   n = length(R)
   for i = 1:n
      hEs[i,i] = precon!(tmp, V, norm(R[i]), R[i], Z[i], z0, innerstab)
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


# ------- Implementation of SimplePairPotential

evaluate!(tmp, V::SimplePairPotential, r::Number, z1, z0) = evaluate!(tmp, V, r)
evaluate_d!(tmp, V::SimplePairPotential, r::Number, z1, z0) = evaluate_d!(tmp, V, r)
evaluate_dd!(tmp, V::SimplePairPotential, r::Number, z1, z0) = evaluate_dd!(tmp, V, r)
evaluate!(tmp, V::SimplePairPotential, r) = evaluate(V, r)
evaluate_d!(tmp, V::SimplePairPotential, r) = evaluate_d(V, r)
evaluate_dd!(tmp, V::SimplePairPotential, r) = evaluate_dd(V, r)
evaluate(V::SimplePairPotential, r::Number) = evaluate!(alloc_temp(V, 1), V, r)
evaluate_d(V::SimplePairPotential, r::Number) = evaluate_d!(alloc_temp_d(V, 1), V, r)
evaluate_dd(V::SimplePairPotential, r::Number) = evaluate_dd!(alloc_temp_dd(V, 1), V, r)

evaluate_d(V::SimplePairPotential, r::Number, R::JVec) =
      (evaluate_d(V, r)/r) * R

function evaluate_dd(V::SimplePairPotential, r::Number, R::JVec)
   dV = evaluate_d(V, r) / r
   ddV = evaluate_dd(V, r)
   return ((ddV - dV)/r^2) * R * R'  + dV * I
end

# ------- Implementation of ExplicitPairPotential

evaluate(p::ExplicitPairPotential, r::Number) = p.f(r)
evaluate_d(p::ExplicitPairPotential, r::Number) = p.f_d(r)
evaluate_dd(p::ExplicitPairPotential, r::Number) = p.f_dd(r)


# ------- Some concrete potentials

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
struct ZeroPairPotential <: SimplePairPotential end

@pot ZeroPairPotential

evaluate(p::ZeroPairPotential, r::T) where {T <: Number} = T(0.0)
evaluate_d(p::ZeroPairPotential, r::T) where {T <: Number} = T(0.0)
evaluate_dd(p::ZeroPairPotential, r::T) where {T <: Number} = T(0.0)
cutoff(p::ZeroPairPotential) = Bool(0) # the weakest number type




# ====================================================================
#   A product of two pair potentials: primarily used for cutoff mechanisms

"product of two `SimplePairPotential`"
mutable struct SimpleProdPot{P1, P2} <: SimplePairPotential
   p1::P1
   p2::P2
end

@pot SimpleProdPot

import Base.*
*(p1::SimplePairPotential, p2::SimplePairPotential) = SimpleProdPot(p1, p2)
evaluate(p::SimpleProdPot, r::Number) = p.p1(r) * p.p2(r)
evaluate_d(p::SimpleProdPot, r::Number) = (p.p1(r) * (@D p.p2(r)) + (@D p.p1(r)) * p.p2(r))
evaluate_dd(p::SimpleProdPot, r::Number) = (p.p1(r) * (@DD p.p2(r)) +
              2 * (@D p.p1(r)) * (@D p.p2(r)) + (@DD p.p1(r)) * p.p2(r))
cutoff(p::SimpleProdPot) = min(cutoff(p.p1), cutoff(p.p2))

"product of two `PairPotential`"
mutable struct ProdPot{P1, P2} <: PairPotential
   p1::P1
   p2::P2
end

@pot ProdPot

import Base.*
*(p1::PairPotential, p2::PairPotential) = ProdPot(p1, p2)
evaluate(p::ProdPot, r::Number, z1, z0) =
      p.p1(r, z1, z0) * p.p2(r, z1, z0)
evaluate_d(p::ProdPot, r::Number, z1, z0) =
      (p.p1(r, z1, z0) * (@D p.p2(r, z1, z0)) + (@D p.p1(r, z1, z0)) * p.p2(r, z1, z0))
evaluate_dd(p::ProdPot, r::Number, z1, z0) =
      (   p.p1(r, z1, z0) * (@DD p.p2(r, z1, z0))
        + 2 * (@D p.p1(r, z1, z0)) * (@D p.p2(r, z1, z0))
        + (@DD p.p1(r, z1, z0)) * p.p2(r, z1, z0)
      )
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
struct WrappedPairPotential <: ExplicitPairPotential
   f::F64fun
   f_d::F64fun
   f_dd::F64fun
   rcut::Float64
end

@pot WrappedPairPotential

cutoff(V::WrappedPairPotential) = V.rcut
# evaluate, etc are all derived from SimplePairPotential

function WrappedPairPotential(V::AnalyticFunction, rcut)
   @assert (0 < rcut < Inf)
   f, f_d, f_dd = let V=V, rc = rcut
      (F64fun(r -> evaluate(V, r) * (r<rc)),
              F64fun(r -> evaluate_d(V, r) * (r<rc)),
              F64fun(r -> evaluate_dd(V, r) * (r<rc)))
   end
   return WrappedPairPotential(f, f_d, f_dd, cutoff(V))
end

function WrappedPairPotential(V::PairPotential)
   @assert (0 < cutoff(V) < Inf)
   f, f_d, f_dd = let V=V, rc = cutoff(V)
      (F64fun(r -> evaluate(V, r) * (r<rc)),
              F64fun(r -> evaluate_d(V, r) * (r<rc)),
              F64fun(r -> evaluate_dd(V, r) * (r<rc)))
   end
   return WrappedPairPotential(f, f_d, f_dd, cutoff(V))
end




# ------------------------------------------------------------------------

# TODO: write more docs + tests for ZBL

"""
Implementation of the ZBL potential to model close approach.
"""
struct ZBLPotential{TV} <: PairPotential
   V::TV   # analytic "inner" potential
end

@pot ZBLPotential

ZBLPotential() = (
   let
      # au = 0.8854 * 0.529 / (Z1^0.23 + Z2^0.23)
      ϵ0 = 0.00552634940621
      C = 1/(4*π*ϵ0)
      E1, E2, E3, E4 = 0.1818, 0.5099, 0.2802, 0.02817
      A1, A2, A3, A4 = 3.2, 0.9423, 0.4028, 0.2016
      V = @analytic(r -> C * (E1*exp(-A1*r) + E2*exp(-A2*r) +
                              E3*exp(-A4*r) + E4*exp(-A4*r) ) / r)
      ZBLPotential(V)
   end)

_zbl_au(Z1, Z2) = (0.8854 * 0.529) / (Z1^0.23 + Z2^0.23)

function evaluate(V::ZBLPotential, r::Number, z1, z0)
   au = _zbl_au(z1, z0)
   return evaluate(V.V, r / au)
end

function evaluate_d(V::ZBLPotential, r::Number, z1, z0)
   au = _zbl_au(z1, z0)
   return evaluate_d(V.V, r / au) / au
end

cutoff(::ZBLPotential) = Inf

write_dict(V::ZBLPotential) = Dict("__id__" => "JuLIP_ZBLPotential")
read_dict(::Val{:JuLIP_ZBLPotential}, D::Dict) = ZBLPotential()
