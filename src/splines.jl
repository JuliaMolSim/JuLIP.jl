#
# implementation of Spline-based pair potentials
# and other functionality
#

import Interpolations 
using Interpolations: interpolate, BSpline, Cubic, Line, OnGrid, 
                      ScaledInterpolation, BSplineInterpolation
using OffsetArrays: OffsetVector
using ForwardDiff


"""
`type SplinePairPotential`

Pair potential defined via splines. In the construction of these splines
atomic units (Angstrom) are assumed. If this is used in different units,
then something will probably break.

### Constructors:
```
SplinePairPotential(xdat, ydat; kwargs...)   # fit spline  from data points
SplinePairPotential(fname; kwargs...)        # load data points from file
```
Keyword arguments:
* `fixcutoff = true`: if true, then the fit will be artificially modified at the cut-off to ensure that the transition to zero is smooth
"""
mutable struct SplinePairPotential{T, SPL} <: SimplePairPotential
   spl::SPL      # The actual spline object
   rcut::T                   # cutoff radius (??? could just use spl.t[end] ???)
end

@pot SplinePairPotential

function SplinePairPotential(xdat::AbstractVector, ydat::AbstractVector, 
                             rcut::Real = xdat[end]; fixcutoff=true)
   # check that the data is equi-spaced. If not, then we need to generalize the 
   # code to allow splines on arbitrary grids (griddedinterpolations?)
   if maximum(abs, diff(diff(xdat))) > 1e-12 
      @show xdat[1:10]
      @show xdat[end-10:end]
      @show maximum(abs, diff(diff(xdat)))
      error("spline knots arent equispaced")
   end
   h = xdat[2] - xdat[1] 
   # add xdat points to the left to make sure zero is included 
   addn = ceil(Int, xdat[1]/h)
   xrg = range(xdat[1] - addn * h, xdat[end]+1.5*h, step=h)
   ydat = [ fill(ydat[1], addn); ydat; [0.0] ]
   if fixcutoff
      # set the last few spline data-points to 0 to get a guaranteed
      # smooth transition to 0 at the cutoff.
      ydat[end-2:end] .= 0.0   # end-order=1:end 
   end
   # construct the spline 
   spl_pre = interpolate(ydat, BSpline(Cubic(Line(OnGrid()))))
   # ... and rescale it to the correct grid 
   spl = Interpolations.scale(spl_pre, xrg)
   return SplinePairPotential{typeof(rcut), typeof(spl)}(spl, rcut)
end


SplinePairPotential(fname::AbstractString; kwargs...) =
      SplinePairPotential(load_ppfile(fname)...; kwargs...)



cutoff(V::SplinePairPotential) = V.rcut

evaluate(V::SplinePairPotential{T}, r::Number) where {T} = 
      r < V.rcut ? V.spl(r) : zero(T)

evaluate_d(V::SplinePairPotential{T}, r::Number) where {T} = 
      r < V.rcut ? Interpolations.gradient(V.spl, r)[1] : zero(T)

evaluate_dd(V::SplinePairPotential{T}, r::Number)  where {T} = 
      r < V.rcut ? Interpolations.hessian(V.spl, r)[1] : zero(T)




function load_ppfile(fname::AbstractString)
   if fname[end-3:end] == ".plt"
      return load_plt(fname)
   else
      error("unknown file type: $(fname[end-3:end])")
   end
   nothing
end


function load_plt(fname::AbstractString)
   data = readdlm(fname, Float64; comments=true)
   return data[:, 1], data[:, 2]
end
