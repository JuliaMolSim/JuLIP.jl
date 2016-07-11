

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
    1.0 ./ ( 1.0 + exp( Lc ./ ( max(Rc-r, 0.0) + _eps_ ) ) )

"derivative of `cutsw`"
@inline function cutsw_d(r, Rc, Lc)
    t = 1 ./ ( max(Rc-r, 0.0) + _eps_ )    # a numerically stable (Rc-r)^{-1}
    e = 1.0 ./ (1.0 + exp(Lc * t))         # compute exponential only once
    return - Lc * (1.0 - e) .* e .* t.^2
end


"""
`type SWCutoff`: SW type cut-off potential with C^∞ regularity
     1.0 / ( 1.0 + exp( Lc / (Rc-r) ) )

This is not very optimised: one could speed up `evaluate_d` significantly
by avoiding multiple evaluations.
"""
type SWCutoff <: AbstractCutoff
    pp::PairPotential
    Rc::Float64
    Lc::Float64
end
@inline evaluate(p::SWCutoff, r) = p.pp(r) .* cutsw(r, p.Rc, p.Lc)
@inline evaluate_d(p::SWCutoff, r) = p.pp(r) .* cutsw_d(r, p.Rc, p.Lc) +
                                       (@D p.pp(r)) .* cutsw(r, p.Rc, p.Lc)
cutoff(p::SWCutoff) = p.Rc


######################## Shift-Cutoff: one should not use this!

"""
`ShiftCutoff` : takes the pair-potential and shifts and truncates it
    f_cut(r) = (f(r) - f(rcut)) .* (r <= rcut)

Default constructor is
```
    ShiftCutoff(Rc, pp)
```
"""
type ShiftCutoff <: AbstractCutoff
    pp::PairPotential
    Rc::Float64
    Jc::Float64
end
ShiftCutoff(pp, Rc) = ShiftCutoff(pp, Rc, pp(Rc))
@inline evaluate(p::ShiftCutoff, r) = (p.pp(r) - p.Jc) .* (r .<= p.Rc)
@inline evaluate_d(p::ShiftCutoff, r) = (@D p.pp(r)) .* (r .<= p.Rc)
cutoff(p::SWCutoff) = p.Rc


######################## Quintic Spline cut-off:


function fcut(r, r0, r1)
    s = 1 - (r-r0) / (r1-r0)
    return ((s .>= 1) + (0 .<= s .< 1) .* (6 * s.^5 - 15 * s.^4 + 10 * s.^3) )
end

"""
Derivative of `cut`; see documentation of `cut`.
"""
function fcut_d(r, r0, r1)
    s = 1-(r-r0) / (r1-r0)
    return ( - (30*s.^4 - 60 * s.^3 + 30 * s.^2) / (r1-r0)
             .* (0 .< s .< 1) )
end

"""
`SplineCutoff` : Piecewise quintic C^{2,1} regular polynomial.

Parameters:

* pp : wrapped pair potential
* r0 : inner cut-off radius
* r1 : outer cut-off radius.
"""
type SplineCutoff
   pp::PairPotential
   r0::Float64
   r1::Float64
end
@inline evaluate(p::SplineCutoff, r) = p.pp(r) .* fcut(r, p.r0, p.r1)
@inline evaluate_d(p::SplineCutoff, r) = p.pp(r) .* fcut_d(r, p.r0, p.r1) +
                                       (@D p.pp(r)) .* fcut(r, p.r0, p.r1)
cutoff(p::SplineCutoff) = p.r1
