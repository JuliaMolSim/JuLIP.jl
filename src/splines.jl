#
# implementation of Spline-based pair potentials
# and other functionality
#
# TODO: explore allowing arbitrary order of splines?
#

import Dierckx
using Dierckx: Spline1D
# https://github.com/kbarbary/Dierckx.jl

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
* `s = 1e-2`: balances smoothness versus fit; basically an error bound, see Dierckx website for more info
* `fixcutoff = true`: if true, then the fit will be artificially modified at the cut-off to ensure that the transition to zero is smooth
* `order = 3`: can use lower or higher order splines (0 <= order <= 5) but only 3 is tested
* `w = (1.0 + ydat).^(-2)`: this gives relative weights to datapoints to ensure a good in the important regions; the intuition behind the default choice is that is prevents overfitting at very high energies which are not physical anyhow, but ensure that sufficient data points are used in the low energy region.
"""
mutable struct SplinePairPotential <: PairPotential
   spl::Spline1D          # The actual spline object
   rcut::Float64          # cutoff radius (??? could just use spl.t[end] ???)
   wrk::Vector{Float64}   # a work array for faster evaluation of derivatives
end

@pot SplinePairPotential

SplinePairPotential(spl::Spline1D) =
   SplinePairPotential(spl, maximum(spl.t), Vector{Float64}(undef, length(spl.t)))



cutoff(V::SplinePairPotential) = V.rcut

evaluate(V::SplinePairPotential, r, R) = evaluate(V, r)
evaluate(V::SplinePairPotential, r) = V.spl(r)

_deriv(V::SplinePairPotential, r, nu) =
   Dierckx.derivative(V.spl, r, nu)
   # Dierckx.__derivative(V.spl.t, V.spl.c, V.spl.k, r, nu, V.spl.bc, V.wrk)

evaluate_d(V::SplinePairPotential, r, R) = _deriv(V, r, 1)
evaluate_d(V::SplinePairPotential, r) = _deriv(V, r, 1)

evaluate_dd(V::SplinePairPotential, r, R) = _deriv(V, r, 2)
evaluate_dd(V::SplinePairPotential, r) = _deriv(V, r, 2)



function SplinePairPotential(xdat, ydat; s = 1e-2, fixcutoff=true, order=3,
                             w = (1.0 .+ abs.(ydat)).^(-2))
   # this creates a "fit" with s determining the balance between smoothness
   # and fitting the data (basically an error bound)
   spl = Spline1D(xdat, ydat; bc = "zero", s = s, k = order, w = w)
   # set the last few spline data-points to 0 to get a guaranteed
   # smooth transition to 0 at the cutoff.
   if fixcutoff
      spl.c[end-order+1:end] .= 0.0
   end
   return SplinePairPotential(spl)
end


SplinePairPotential(fname::AbstractString; kwargs...) =
   SplinePairPotential(load_ppfile(fname)...; kwargs...)



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
