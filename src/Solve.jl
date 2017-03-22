"""
`module JuLIP.Solve`

Contains a few geometry optimisation routines, for now see
the help for these:

* `minimise!`
"""
module Solve

import Optim
import LineSearches

using Optim: OnceDifferentiable, optimize, ConjugateGradient, LBFGS

using JuLIP: AbstractAtoms, Preconditioner, update!, Identity,
            dofs, energy, gradient, set_dofs!, set_constraint!, site_energies, Dofs

using JuLIP.Potentials: SitePotential


export minimise!



Ediff(V::SitePotential, at::AbstractAtoms, Es0::Vector{Float64}) =
   sum_kbn(site_energies(V, at) - Es0)

Ediff(at::AbstractAtoms, Es0::Vector{Float64}, x::Dofs) =
   Ediff(calculator(at), set_dofs!(at, x), Es0)


"""
`minimise!(at::AbstractAtoms)`: geometry optimisation

`at` must have a calculator and a constraint attached.

## Keyword arguments:
* `precond = Identity()` : preconditioner
* `grtol = 1e-6` : gradient tolerance (max-norm)
* `ftol = 1e-32` : objective tolerance
* `Optimiser = :auto`, `:auto` should always work, at least on the master
   branch of `Optim`; `:lbfgs` needs the `extraplbfgs2` branch, which is not
   yet merged. Other options might be introduced in the future.
* `verbose = 1`: 0 : no output, 1 : final, 2 : iteration and final
"""
function minimise!( at::AbstractAtoms;
                  precond = Identity(), gtol=1e-6, ftol=1e-32,
                  method = :auto,
                  verbose = 1,
                  robust_energy_difference = false )

   # create an objective function
   if robust_energy_difference
      Es0 = site_energies(at)
      objective = OnceDifferentiable(
         x -> Ediff(at, Es0, x),
         (x, g) -> copy!(g, gradient(at, x))
      )
   else
      objective = OnceDifferentiable(
         x->energy(at, x),
         (x,g)->copy!(g, gradient(at, x))
      )
   end
   # call Optim.jl
   # TODO: use verb flag to determine whether detailed output is wanted
   if method == :auto
      if isa(precond, Identity)
         optimiser = Optim.ConjugateGradient()
      else
         optimiser = Optim.ConjugateGradient( P = precond,
                           precondprep = (P, x) -> update!(P, at, x),
                           linesearch = LineSearches.bt2!)
      end
   elseif method == :lbfgs
      optimiser = Optim.LBFGS( P = precond, extrapolate=true,
                        precondprep = (P, x) -> update!(P, at, x),
                        linesearch = LineSearches.bt2! )
   else
      error("JulIP.Solve.minimise!: unkonwn `method` option")
   end

   results = optimize( objective, dofs(at), optimiser,
                        Optim.Options(f_tol = ftol, g_tol = gtol, show_trace = (verbose > 1)) )
   set_dofs!(at, Optim.minimizer(results))
   # analyse the results
   if verbose > 0
      println(results)
   end
   return results
end


# saddle search
# include("saddlesearch.jl")


end
