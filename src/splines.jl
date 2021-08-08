#
# implementation of Spline-based pair potentials
# and other functionality
#

import Dierckx
using Dierckx: Spline1D
using ForwardDiff

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
* `s = 0`: balances smoothness versus fit; by default this matches the fits obtained from ASE and LAMMPS
* `fixcutoff = true`: if true, then the fit will be artificially modified at the cut-off to ensure that the transition to zero is smooth
* `order = 3`: can use lower or higher order splines (0 <= order <= 5) but only 3 is tested
* `w = (1.0 + ydat).^(-2)`: this gives relative weights to datapoints to ensure a good in the important regions; the intuition behind the default choice is that is prevents overfitting at very high energies which are not physical anyhow, but ensure that sufficient data points are used in the low energy region.
"""
mutable struct SplinePairPotential <: SimplePairPotential
   spl::Spline1D          # The actual spline object
   rcut::Float64          # cutoff radius (??? could just use spl.t[end] ???)
   wrk::Vector{Vector{Float64}}  # work array for faster evaluation of derivatives
end

@pot SplinePairPotential

SplinePairPotential(spl::Spline1D) =
   SplinePairPotential(spl, maximum(spl.t), 
                 [ Vector{Float64}(undef, length(spl.t)) for _=1:JuLIP.nthreads() ] )


cutoff(V::SplinePairPotential) = V.rcut

evaluate(V::SplinePairPotential, r::Number) = V.spl(r)
evaluate_d(V::SplinePairPotential, r::Number) = 
         Dierckx._derivative(V.spl.t, V.spl.c, V.spl.k, r, 1, V.spl.bc, _get_wrk(V))
evaluate_dd(V::SplinePairPotential, r::Number) = 
         Dierckx._derivative(V.spl.t, V.spl.c, V.spl.k, r, 2, V.spl.bc, _get_wrk(V))

# a few interface routines so we can define AD in a nice way.
_evalspl(s::Spline1D, r) = s(r)
_evalspl_d(s::Spline1D, r) = Dierckx.derivative(s, r, 1)
_evalspl_dd(s::Spline1D, r) = Dierckx.derivative(s, r, 2)

function _evalspl(s::Spline1D, d::ForwardDiff.Dual{T}) where T
   x = ForwardDiff.value(d)
   ForwardDiff.Dual{T}( _evalspl(s, x),
                        _evalspl_d(s, x) * ForwardDiff.partials(d) )
end

function _evalspl_d(s::Spline1D, d::ForwardDiff.Dual{T}) where T
   x = ForwardDiff.value(d)
   ForwardDiff.Dual{T}( _evalspl_d(s, x),
                        _evalspl_dd(s, x) * ForwardDiff.partials(d) )
end

function _get_wrk(s::SplinePairPotential)
   tid = threadid() 
   while tid < length(s.wrk)
      push!(s.wrk, Vector{Float64}(undef, length(spl.t)))
   end 
   return s.wrk[tid]
end

evaluate!(wrk, V::SplinePairPotential, r::Number) = V(r)

evaluate_d!(wrk, V::SplinePairPotential, r::Number) = 
      Dierckx._derivative(V.spl.t, V.spl.c, V.spl.k, r, 1, V.spl.bc, wrk)

evaluate_dd!(wrk, V::SplinePairPotential, r::Number) = 
      Dierckx._derivative(V.spl.t, V.spl.c, V.spl.k, r, 2, V.spl.bc, wrk)

function evaluate!(wrk, V::SplinePairPotential, d::ForwardDiff.Dual{T}) where T
   x = ForwardDiff.value(d)
   val = evaluate!(wrk, V::SplinePairPotential, x)
   dval = evaluate_d!(wrk, V::SplinePairPotential, x)
   ForwardDiff.Dual{T}(val, dval * ForwardDiff.partials(d))
end

function evaluate_d!(wrk, V::SplinePairPotential, d::ForwardDiff.Dual{T}) where T
   x = ForwardDiff.value(d)
   val = evaluate_d!(wrk, V::SplinePairPotential, x)
   dval = evaluate_dd!(wrk, V::SplinePairPotential, x)
   ForwardDiff.Dual{T}(val, dval * ForwardDiff.partials(d))
end


function SplinePairPotential(xdat, ydat; s = 0, fixcutoff=true, order=3,
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
