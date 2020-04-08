
import Base.*

export SWCutoff, SplineCutoff, Shift, HS, C0Shift, C1Shift, C2Shift

# this file is include from Potentials.jl
# i.e. it is part of JuLIP.Potentials
# it implements three types of cutoff potentials.


######################## Stillinger-Weber type cutoff


"""
`cutsw(r, Rc, Lc)`

Implementation of the C^∞ Stillinger-Weber type cut-off potential
    1.0 / ( 1.0 + exp( lc / (Rc-r) ) )

`d_cutinf` implements the first derivative
"""
@inline cutsw(r::T, Rc, Lc) where {T} =
    T(1) / ( T(1) + exp( Lc / ( max(Rc-r, T(0)) + T(1e-2) ) ) )

"derivative of `cutsw`"
@inline function cutsw_d(r::T, Rc, Lc) where {T}
    t = T(1) / ( max(Rc-r, T(0)) + T(1e-2) )    # a numerically stable (Rc-r)^{-1}
    e = T(1) / (T(1) + exp(Lc * t))             # compute exponential only once
    return - Lc * (T(1) - e) * e * t^2
end



"""
`type SWCutoff`: SW type cut-off potential with C^∞ regularity
     1.0 / ( 1.0 + exp( Lc / (Rc-r) ) )

This is not very optimised: one could speed up `evaluate_d` significantly
by avoiding multiple evaluations.

### Parameters

* `Rc` : cut-off radius
* `Lc` : scale
"""
struct SWCutoff{T} <: SimplePairPotential
    Rc::T
    Lc::T
    e0::T
end

@pot SWCutoff

evaluate(p::SWCutoff, r::Number) = p.e0 * cutsw(r, p.Rc, p.Lc)
evaluate_d(p::SWCutoff, r::Number) = p.e0 * cutsw_d(r, p.Rc, p.Lc)
cutoff(p::SWCutoff) = p.Rc

# simplified constructor to ensure compatibility
SWCutoff(Rc, Lc) = SWCutoff(Rc, Lc, one(typeof(Rc)))

# kw-constructor (original SW parameters, pair term)
SWCutoff(; Rc=1.8, Lc=1.0, e0=1.0) = SWCutoff(Rc, Lc, e0)



######################## Shift-Cutoff:

"""
`Shift{ORD}` : a shift-cutoff function

It is not technically a cut-off function but a cut-off operator. Let
`f(r)` with real arguments and range, then
```
g = f * Shift(o, rcut)
```
creates a new function `g(r) = (f(r) - p(r)) * (r < rcut)` where
`p(r)` is a polynomial of order `o` such that all derivatives of `g` up
to order `o` vanish at `r = rcut`. There order must be an integer
between -1 and 2. Choosing `o=-1` creates a Heaviside function.

### Equivalent constructors:
```
HS(r::Float64) = Shift(-1, r)
C0Shift(r::Float64) = Shift(0, r)
C1Shift(r::Float64) = Shift(1, r)
C2Shift(r::Float64) = Shift(2, r)
```

### Example:
```
lj = LennardJones()  # standard lennad-jones potential
V = lj * C2Shift(2.5)
```
Now `V` is a C2-continuous `PairPotential` with support (0, 2.5]."""
struct Shift{ORD, TV, T} <: SimplePairPotential
   ord::Val{ORD}
   V::TV
   rcut::T
   Vcut::T
   dVcut::T
   ddVcut::T
end

@pot Shift


const HS{TV} = Shift{-1, TV}
const C0Shift{TV} = Shift{0, TV}
const C1Shift{TV} = Shift{1, TV}
const C2Shift{TV} = Shift{2, TV}

cutoff(p::Shift) = p.rcut

# the basic constructors
"see documentation for `Shift`"
HS(r::AbstractFloat) = Shift(-1, r)
"see documentation for `Shift`"
C0Shift(r::AbstractFloat) = Shift(0, r)
"see documentation for `Shift`"
C1Shift(r::AbstractFloat) = Shift(1, r)
"see documentation for `Shift`"
C2Shift(r::AbstractFloat) = Shift(2, r)
Shift(o::Int, r::AbstractFloat) = Shift(Val(o), nothing, r, 0.0, 0.0, 0.0)
Shift(V, p::Shift{-1}) = Shift(p.ord, V, p.rcut, 0.0, 0.0, 0.0)
Shift(V, p::Shift{0}) = Shift(p.ord, V, p.rcut, V(p.rcut), 0.0, 0.0)
Shift(V, p::Shift{1}) = Shift(p.ord, V, p.rcut, V(p.rcut), (@D V(p.rcut)), 0.0)
Shift(V, p::Shift{2}) = Shift(p.ord, V, p.rcut, V(p.rcut), (@D V(p.rcut)), (@DD V(p.rcut)))
*(V::SimplePairPotential, p::Shift{ORD, Nothing}) where {ORD} = Shift(V, p)
*(p::Shift{ORD, Nothing}, V::SimplePairPotential) where {ORD} = Shift(V, p)


@inline evaluate(p::Shift{-1}, r::Number) = r < p.rcut ? p.V(r) : 0.0
@inline evaluate_d(p::Shift{-1}, r::Number) = r < p.rcut ? (@D p.V(r)) : 0.0
@inline evaluate_dd(p::Shift{-1}, r::Number) = r < p.rcut ? (@DD p.V(r)) : 0.0

@inline evaluate(p::Shift{0}, r::Number) = r < p.rcut ? (p.V(r) - p.Vcut) : 0.0
@inline evaluate_d(p::Shift{0}, r::Number) = r < p.rcut ? (@D p.V(r)) : 0.0
@inline evaluate_dd(p::Shift{0}, r::Number) = r < p.rcut ? (@DD p.V(r)) : 0.0

@inline evaluate(p::Shift{1}, r::Number) = r >= p.rcut ? 0.0 :
      (p.V(r) - p.Vcut - p.dVcut * (r - p.rcut))
@inline evaluate_d(p::Shift{1}, r::Number) = r < p.rcut ? ((@D p.V(r)) - p.dVcut) : 0.0
@inline evaluate_dd(p::Shift{1}, r::Number) = r < p.rcut ? (@DD p.V(r)) : 0.0

@inline evaluate(p::Shift{2}, r::Number) = r >= p.rcut ? 0.0 :
   (p.V(r) - p.Vcut - p.dVcut * (r - p.rcut) - 0.5 * p.ddVcut * (r-p.rcut)^2)
@inline evaluate_d(p::Shift{2}, r::Number) = r >= p.rcut ? 0.0 :
   ((@D p.V(r)) - p.dVcut - p.ddVcut * (r - p.rcut))
@inline evaluate_dd(p::Shift{2}, r::Number) = r >= p.rcut ? 0.0 :
   ((@DD p.V(r)) - p.ddVcut)




######################## Quintic Spline cut-off:

# TODO: switch to general types ?

@inline function fcut(r, r0, r1)
    s = 1.0 - (r-r0) / (r1-r0)
    return (s >= 1.0) + (0.0 < s < 1.0) * (@fastmath 6.0 * s^5 - 15.0 * s^4 + 10.0 * s^3)
end

"Derivative of `fcut`; see documentation of `fcut`."
@inline function fcut_d(r, r0, r1)
    s = 1-(r-r0) / (r1-r0)
    return -(0 < s < 1) * (@fastmath (30*s^4 - 60 * s^3 + 30 * s^2) / (r1-r0))
end

"Second order derivative of `fcut`; see documentation of `fcut`."
function fcut_dd(r, r0, r1)
  s = 1-(r-r0) / (r1-r0)
  return (0 < s < 1) * (@fastmath (120*s^3 - 180 * s^2 + 60 * s) / (r1-r0)^2)
end

"""
`SplineCutoff` : Piecewise quintic C^{2,1} regular polynomial.

Parameters:

* r0 : inner cut-off radius
* r1 : outer cut-off radius.
"""
mutable struct SplineCutoff <: SimplePairPotential
   r0::Float64
   r1::Float64
end

@pot SplineCutoff


@inline evaluate(p::SplineCutoff, r::Number) = fcut(r, p.r0, p.r1)
@inline evaluate_d(p::SplineCutoff, r::Number) = fcut_d(r, p.r0, p.r1)
evaluate_dd(p::SplineCutoff, r::Number) = fcut_dd(r, p.r0, p.r1)
cutoff(p::SplineCutoff) = p.r1
Base.string(p::SplineCutoff) = "SplineCutoff(r0=$(p.r0), r1=$(p.r1))"



# ============ DRAFT: cos-based cutoff ============

@inline function coscut(r::T, r0, r1) where {T <: AbstractFloat}
   if r <= r0
      return T(1.0)
   elseif r > r1
      return T(0.0)
   else
      return @fastmath T(0.5) * (cos( π * (r-r0) / (r1-r0) ) + T(1.0))
   end
end

@inline function coscut_d(r::T, r0, r1) where {T <: AbstractFloat}
   if r0 < r < r1
      return @fastmath - π/(2*(r1-r0)) * sin( π * (r-r0) / (r1-r0) )
   else
      return T(0.0)
   end
end
