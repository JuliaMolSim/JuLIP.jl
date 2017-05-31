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
# using LineSearches: BackTracking

using JuLIP: AbstractAtoms, Preconditioner, update!, Identity,
            dofs, energy, gradient, set_dofs!, set_constraint!, site_energies,
            Dofs, calculator, constraint, AbstractCalculator

using JuLIP.Potentials: SitePotential
using JuLIP.Preconditioners: Exp
using JuLIP.Constraints: FixedCell


export minimise!



Ediff(V::AbstractCalculator, at::AbstractAtoms, Es0::Vector{Float64}) =
   sum_kbn(site_energies(V, at) - Es0)

Ediff(at::AbstractAtoms, Es0::Vector{Float64}, x::Dofs) =
   Ediff(calculator(at), set_dofs!(at, x), Es0)


"""
`minimise!(at::AbstractAtoms)`: geometry optimisation

`at` must have a calculator and a constraint attached.

## Keyword arguments:
* `precond = :auto` : preconditioner; more below
* `grtol = 1e-6` : gradient tolerance (max-norm)
* `ftol = 1e-32` : objective tolerance
* `Optimiser = :auto`, `:auto` should always work, at least on the master
   branch of `Optim`; `:lbfgs` needs the `extraplbfgs2` branch, which is not
   yet merged. Other options might be introduced in the future.
* `verbose = 1`: 0 : no output, 1 : final, 2 : iteration and final

## Preconditioner

`precond` may be a valid preconditioner, e.g.,
`Identity()` or `Exp(at)`, or one of the following symbols

* `:auto` : the code will make the best choice it can with the avilable
   information
* `:exp` : will use `Exp(at)`
* `:id` : will use `Identity()`
"""
function minimise!(at::AbstractAtoms;
                  precond = :auto,
                  method = :auto,
                  gtol=1e-5, ftol=1e-32,
                  verbose = 1,
                  robust_energy_difference = false )

   # create an objective function
   if robust_energy_difference
      Es0 = site_energies(at)
      obj_f = x -> Ediff(at, Es0, x)
   else
      obj_f = x->energy(at, x)
   end
   obj_g! = (x, g) -> copy!(g, gradient(at, x))    # switch to (g, x) for Optim 0.8+

   # create a preconditioner
   if isa(precond, Symbol)
      if precond == :auto
         if isa(constraint(at), FixedCell)
            precond = :exp
         else
            precond = :id
         end
      end
      if precond == :exp
         if method == :lbfgs
            precond = Exp(at, energyscale = :auto)
         else
            precond = Exp(at)
         end
      elseif precond == :id
         precond = Identity()
      else
         error("unknown symbol for precond")
      end
   end

   # choose the optimisation method Optim.jl
   if method == :auto
      if isa(precond, Identity)
         optimiser = Optim.ConjugateGradient()
      else
         optimiser = Optim.ConjugateGradient( P = precond,
                           precondprep = (P, x) -> update!(P, at, x),
                           linesearch = LineSearches.bt2! )    # LineSearches.BackTracking(order=2)
      end
   elseif method == :lbfgs
      optimiser = Optim.LBFGS( P = precond, extrapolate=true,
                        precondprep = (P, x) -> update!(P, at, x),
                        linesearch = LineSearches.bt2! )        # BackTracking(order=2)
   else
      error("JulIP.Solve.minimise!: unknown `method` option")
   end

   results = optimize( obj_f, obj_g!, dofs(at), optimiser,
                        Optim.Options( f_tol = ftol, g_tol = gtol,
                                       show_trace = (verbose > 1)) )
   set_dofs!(at, Optim.minimizer(results))
   # analyse the results
   if verbose > 0
      println(results)
   end
   return results
end



end
