

export SWCutoff, ShiftCutoff, SplineCutoff



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
@inline cutsw(r, Rc, Lc) =
    1.0 ./ ( 1.0 + exp( Lc ./ ( max(Rc-r, 0.0) + 1e-2 ) ) )

"derivative of `cutsw`"
@inline function cutsw_d(r, Rc, Lc)
    t = 1 ./ ( max(Rc-r, 0.0) + 1e-2 )    # a numerically stable (Rc-r)^{-1}
    e = 1.0 ./ (1.0 + exp(Lc * t))         # compute exponential only once
    return - Lc * (1.0 - e) .* e .* t.^2
end


# doc below
@pot type SWCutoff <: PairPotential
    Rc::Float64
    Lc::Float64
    e0::Float64
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
SWCutoff

evaluate(p::SWCutoff, r) = p.e0 * cutsw(r, p.Rc, p.Lc)
evaluate_d(p::SWCutoff, r) = p.e0 * cutsw_d(r, p.Rc, p.Lc)
cutoff(p::SWCutoff) = p.Rc

# TODO: hack to make TB work (reconsider this)
evaluate(p::SWCutoff, r, R) = evaluate(p, r)
evaluate_d(p::SWCutoff, r, R) = evaluate_d(p, r)

# simplified constructor to ensure compatibility
SWCutoff(Rc, Lc) = SWCutoff(Rc, Lc, 1.0)

# kw-constructor (original SW parameters, pair term)
SWCutoff(; Rc=1.8, Lc=1.0, e0=1.0) = SWCutoff(Rc, Lc, e0)



######################## Shift-Cutoff: one should not use this!
# TODO: why is this commented out? probably uncomment and check.

# """
# `ShiftCutoff` : takes the pair-potential and shifts and truncates it
#     f_cut(r) = (f(r) - f(rcut)) .* (r <= rcut)
#
# Note this is not constructed by multiplying the cutoff potential with the
# actual potential, but it is constructed by
# ```julia
#     ShiftCutoff(Rc, pp)
# ```
#
# **WARNING:** this creates potentials that are not even C^1, which can
# cause all sorts of hell.
# """
# @pot type ShiftCutoff{T <: PairPotential} <: PairPotential
#     pp::T
#     Rc::Float64
#     Jc::Float64
# end
# ShiftCutoff(pp, Rc) = ShiftCutoff(pp, Rc, pp(Rc))
# evaluate(p::ShiftCutoff, r) = (p.pp(r) - p.Jc) .* (r .<= p.Rc)
# evaluate_d(p::ShiftCutoff, r) = (@D p.pp(r)) .* (r .<= p.Rc)
# cutoff(p::SWCutoff) = p.Rc



######################## Quintic Spline cut-off:

function fcut(r, r0, r1)
    s = 1 - (r-r0) / (r1-r0)
    return ((s .>= 1) + (0 .<= s .< 1) .* (6 * s.^5 - 15 * s.^4 + 10 * s.^3) )
end

"Derivative of `fcut`; see documentation of `fcut`."
function fcut_d(r, r0, r1)
    s = 1-(r-r0) / (r1-r0)
    return ( - (30*s.^4 - 60 * s.^3 + 30 * s.^2) / (r1-r0)
             .* (0 .< s .< 1) )
end
"Second order derivative of `fcut`; see documentation of `fcut`."
function fcut_dd(r, r0, r1)
  s = 1-(r-r0) / (r1-r0)
  return ( (120*s.^3 - 180 * s.^2 + 60 * s) / (r1-r0)^2
            .*(0 .< s .< 1))
end

@pot type SplineCutoff <: PairPotential
   r0::Float64
   r1::Float64
end

"""
`SplineCutoff` : Piecewise quintic C^{2,1} regular polynomial.

Parameters:

* r0 : inner cut-off radius
* r1 : outer cut-off radius.
"""
SplineCutoff

evaluate(p::SplineCutoff, r) = fcut(r, p.r0, p.r1)
evaluate_d(p::SplineCutoff, r) = fcut_d(r, p.r0, p.r1)
evaluate_dd(p::SplineCutoff, r) = fcut_dd(r, p.r0, p.r1)
cutoff(p::SplineCutoff) = p.r1
Base.string(p::SplineCutoff) = "SplineCutoff(r0=$(p.r0), r1=$(p.r1))"
