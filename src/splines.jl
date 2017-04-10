#
# implementation of Spline-based pair potentials
# and other functionality
#
# TODO: explore allowing arbitrary order of splines?
#

using Dierckx: derivative, Spline1D
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
end

cutoff(V::SplinePairPotential) = V.rcut

evaluate(V::SplinePairPotential, r, R) = V.spl(r)

evaluate(V::SplinePairPotential, r, R) = V.spl(r)


"""
`type SplinePairPotential`

Pair potential defined via splines. In the construction of these splines
atomic units (Angstrom) are assumed. If this is used in different units,
then something will surely break.
"""
SplinePairPotential

function SplinePairPotential(xdat, ydat, xknots::AbstractVector;
                             fixcutoff = true, kwargs...)
   xknots = collect(xknots)
   if fixcutoff
      # add the cutoff to the knots
      rcut = maximum(xdat)
      push!(xknots, rcut)
      # add two extra points to ensure the potential becomes zero near the cutoff
      append!(xknots, [rcut + 1e-2, rcut + 2e-2])
      # add data points to ensure the fit is correct between rcut and rcut+2e-2
      append!(xdat, [rcut + j*1e-3 for j = 1:30])
      append!(ydat, zeros(30))
   end
   # construct the spline
   spl = Spline1D(xdat, ydat, xknots; bc = "zero", kwargs...)
   return SplinePairPotential(spl, maximum(xknots))
end


# function SplinePairPotential(xdat, ydat, nknots::Integer; kwargs...)
# end


function SplinePairPotential(fname::AbstractString, nknots::Integer; kwargs...)
   xdat, ydat = load_ppfile(fname)

end


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
