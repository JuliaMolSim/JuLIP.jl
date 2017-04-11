#
# implementation of Spline-based pair potentials
# and other functionality
#
# TODO: explore allowing arbitrary order of splines?
#

using Dierckx: Spline1D
# https://github.com/kbarbary/Dierckx.jl

function spline_interpolate(x, y; order=3)
   @assert order == 3
   return Spline1D(x, y)
end


function spline_fit(xdat, ydat, xknots; order = 3, kwargs...)
   @assert order == 3
   return Spline1D(xdat, ydat, xknots; kwargs...)
end


@pot type SplinePairPotential <: PairPotential
   spl::Spline1D
   rcut::Float64
   wrk::
end

cutoff(V::SplinePairPotential) = V.rcut

evaluate(V::SplinePairPotential, r, R) = V.spl(r)

_deriv(V::SplinePairPotential, r, nu) =
   Dierckx.__derivative(V.spl.t, V.spl.c, V.spl.k, r, nu, V.spl.bc, V.wrk)

evaluate_d(V::SplinePairPotential, r, R) = _deriv(V, r, 1)

evaluate_dd(V::SplinePairPotential, r, R) = _deriv(V, r, 2)


"""
`type SplinePairPotential`

Pair potential defined via splines. In the construction of these splines
atomic units (Angstrom) are assumed. If this is used in different units,
then something will surely break.
"""
SplinePairPotential


function SplinePairPotential(xdat, ydat; s = 1e-2, fixcutoff=true, order=3,
                             w = (1.0 + ydat).^(-2))
   @assert order == 3
   # TODO: explore whether it is worthwhile using weights in this construction?
   spl = Spline1D(xdat, ydat; bc = "zero", s = s, k = order, w = w)
   # set the last few spline data-points to 0 to get a guaranteed
   # smooth transition to 0 at the cutoff.
   if fixcutoff
      spl.c[end-order+1:end] = 0.0
   end
   return SplinePairPotential(spl, maximum(splt.t))
end


SplinePairPotential(fname::AbstractString; kwargs...) =
   SplinePairPotential(load_ppfile(fname)...; kwargs...)



function load_ppfile(fname::AbstractString)
   if fname[end-3:end] == "plt"
      return load_plt(fname)
   else
      error("unknown file type")
   end
   nothing
end


function load_plt(fname::AbstractString)
   data = readdlm(fname, Float64)
   return data[:, 1], data[:, 2]
end
