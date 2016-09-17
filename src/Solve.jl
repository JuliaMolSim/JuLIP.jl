"""
`module JuLIP.Solve`

Contains a few geometry optimisation routines, for now see
the help for these:

* `minimise`
"""
module Solve


import Optim
import Optim: DifferentiableFunction, optimize, ConjugateGradient, LBFGS


using JuLIP: AbstractAtoms, Preconditioner, update!, Identity, update!,
            dofs, energy, grad


export minimise!


"""
`minimise!(at::AbstractAtoms)`: geometry optimisation

`at` must have a calculator and a constraint attached.

## Keyword arguments:
* `precond = Identity()` : preconditioner
* `grtol = 1e-6`
* `ftol = 1e-32`
* `Optimiser = Optim.ConjugateGradient`
* `verbose = 0`: 0 : no output, 1 : final, 2 : iteration
"""
function minimise!( at::AbstractAtoms;
                  precond = Identity(), gtol=1e-6, ftol=1e-32,
                  Optimiser = ConjugateGradient,
                  verbose = 1 )

   # create an objective function
   objective = DifferentiableFunction( x->energy(at, x),
                                       (x,g)->copy!(g, grad(at, x)) )
   # call Optim.jl
   # TODO: use verb flag to determine whether detailed output is wanted
   if isa(precond, Identity)
      optimiser = Optim.ConjugateGradient(P = precond,
                               precondprep! = (P, x) -> update!(P, at, x) )
   else
      optimiser = Optim.LBFGS( P = precond,
                        precondprep! = (P, x) -> update!(P, at, x),
                        linesearch! = Optim.quadbt_linesearch! )
   end
   results = optimize( objective, dofs(at), method = optimiser,
                        f_tol = ftol, g_tol = gtol, show_trace = (verbose > 1) )
   # analyse the results
   if verbose > 0
      println(results)
   end
   return results
end




# saddle search
# include("saddlesearch.jl")




end
