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
using LineSearches: BackTracking

using LinearAlgebra: I
using JuLIP: AbstractAtoms, update!,
             dofs, energy, gradient, set_dofs!, site_energies,
             Dofs, calculator, AbstractCalculator, r_sum, fixedcell

using JuLIP.Potentials: SitePotential
using JuLIP.Preconditioners: Exp



export minimise!



Ediff(V::AbstractCalculator, at::AbstractAtoms, Es0::Vector) =
   r_sum(site_energies(V, at) - Es0)

Ediff(at::AbstractAtoms, Es0::Vector, x::Dofs) =
   Ediff(calculator(at), set_dofs!(at, x), Es0)


"""
`minimise!(at::AbstractAtoms)`: geometry optimisation

`at` must have a calculator and a constraint attached.

## Keyword arguments:
* `precond = :auto` : preconditioner; more below
* `gtol = 1e-6` : gradient tolerance (max-norm)
* `ftol = 1e-32` : objective tolerance
* `Optimiser = :auto`, `:auto` should always work, at least on the master
   branch of `Optim`; `:lbfgs` needs the `extraplbfgs2` branch, which is not
   yet merged. Other options might be introduced in the future.
* `verbose = 1`: 0 : no output, 1 : final, 2 : iteration and final
* `robust_energy_difference = false` : if true use Kahan summation of site energies
* `store_trace = false` : store history of energy and norm of forces
* `extended_trace = false`: also store full history of postions and forces
* `maxstep = Inf`: maximum step size, useful if initial gradient is very large
* `callback`: callback function to pass to `optimize()`, e.g. to use alternate convergence criteria

## Preconditioner

`precond` may be a valid preconditioner, e.g., `I` or `Exp(at)`, or one of
the following symbols

* `:auto` : the code will make the best choice it can with the avilable
   information
* `:exp` : will use `Exp(at)`
* `:id` : will use `I`
"""
function minimise!(at::AbstractAtoms;
                  precond = :auto,
                  method = :auto,
                  gtol=1e-5, ftol=1e-32,
                  verbose = 1,
                  robust_energy_difference = false,
                  store_trace = false,
                  extended_trace = false,
                  maxstep = Inf,
                  callback = nothing,
                  g_calls_limit = 1_000)

   # create an objective function
   if robust_energy_difference
      Es0 = site_energies(at)
      obj_f = x -> Ediff(at, Es0, x)
   else
      obj_f = x->energy(at, x)
   end
   obj_g! = (g, x) -> copyto!(g, gradient(at, x))

   # create a preconditioner
   if isa(precond, Symbol)
      if precond == :auto
         if fixedcell(at)
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
         precond = I
      else
         error("unknown symbol for precond")
      end
   end

   # choose the optimisation method Optim.jl
   if method == :auto || method == :cg
      if precond == I
         optimiser = ConjugateGradient(linesearch = BackTracking(order=2, maxstep=maxstep))
      else
         optimiser = ConjugateGradient( P = precond,
                           precondprep = (P, x) -> update!(P, at, x),
                           linesearch = BackTracking(order=2, maxstep=maxstep) )
      end
   elseif method == :lbfgs
      optimiser = LBFGS( P = precond,
                        precondprep = (P, x) -> update!(P, at, x),
                        alphaguess = LineSearches.InitialHagerZhang(),
                        linesearch = BackTracking(order=2, maxstep=maxstep) )
   elseif method == :sd
      optimiser = Optim.GradientDescent( P = precond,
                  precondprep = (P, x) -> update!(P, at, x),
                  linesearch = BackTracking(order=2, maxstep=maxstep) )
   else
      error("JuLIP.Solve.minimise!: unknown `method` option")
   end

   results = optimize( obj_f, obj_g!, dofs(at), optimiser,
                        Optim.Options( f_tol = ftol, g_tol = gtol,
                                       g_calls_limit = g_calls_limit,
                                       store_trace = store_trace,
                                       extended_trace = extended_trace,
                                       callback = callback,
                                       show_trace = (verbose > 1)) )
   set_dofs!(at, Optim.minimizer(results))
   # analyse the results
   if verbose > 0
      println(results)
   end
   return results
end


end
