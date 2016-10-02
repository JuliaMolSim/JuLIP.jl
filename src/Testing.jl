

"""
This module supplies some functions for testing of the implementation.
Look at `?` for

* `fdtest` : general finite-difference test
* `fdtest_R2R`: fd test for F : ℝ → ℝ
"""
module Testing

using JuLIP: AbstractCalculator, AbstractAtoms, energy, gradient, forces,
         constraint, calculator, set_positions!, dofs, NullConstraint,
         mat, vecs, positions, rattle!, set_dofs!, set_constraint!, set_calculator!
using JuLIP.Potentials: PairPotential, evaluate, evaluate_d, grad, @D
using JuLIP.Constraints: FixedCell


export fdtest

"""
generic finite-difference test for scalar F

* `fdtest(F::Function, dF::Function, x)`

* `fdtest(V::PairPotential, r::Vector{Float64})`

TODO: complete documentation
"""
function fdtest(F::Function, dF::Function, x; verbose=true)
   errors = Float64[]
   E = F(x)
   dE = dF(x)
   # loop through finite-difference step-lengths
   @printf("---------|----------- \n")
   @printf("    h    | error \n")
   @printf("---------|----------- \n")
   for p = 2:11
      h = 0.1^p
      dEh = copy(dE)
      for n = 1:length(dE)
         x[n] += h
         dEh[n] = (F(x) - E) / h
         x[n] -= h
      end
      push!(errors, vecnorm(dE - dEh, Inf))
      @printf(" %1.1e | %4.2e  \n", h, errors[end])
   end
   @printf("---------|----------- \n")
   if minimum(errors) <= 1e-3 * maximum(errors)
      println("passed")
      return true
   else
      warn("""It seems the finite-difference test has failed, which indicates
      that there is an inconsistency between the function and gradient
      evaluation. Please double-check this manually / visually. (It is
      also possible that the function being tested is poorly scaled.)""")
      return false
   end
end

"finite-difference test for a function V : ℝ → ℝ"
function fdtest_R2R(F::Function, dF::Function, x::Vector{Float64};
                     verbose=true)
   errors = Float64[]
   E = [ F(t) for t in x ]
   dE = [ dF(t) for t in x ]
   # loop through finite-difference step-lengths
   if verbose
      @printf("---------|----------- \n")
      @printf("    h    | error \n")
      @printf("---------|----------- \n")
   end
   for p = 2:11
      h = 0.1^p
      dEh = ([F(t+h) for t in x ] - E) / h
      push!(errors, vecnorm(dE - dEh, Inf))
      if verbose
         @printf(" %1.1e | %4.2e  \n", h, errors[end])
      end
   end
   if verbose
      @printf("---------|----------- \n")
   end
   if minimum(errors) <= 1e-3 * maximum(errors[1:2])
      println("passed")
      return true
   else
      warn("""is seems the finite-difference test has failed, which indicates
            that there is an inconsistency between the function and gradient
            evaluation. Please double-check this manually / visually. (It is
            also possible that the function being tested is poorly scaled.)""")
      return false
   end
end


fdtest(V::PairPotential, r::AbstractVector; kwargs...) =
               fdtest_R2R(s -> V(s), s -> (@D V(s)), collect(r); kwargs...)



function fdtest(calc::AbstractCalculator, at::AbstractAtoms;
                verbose=true, rattle=0.01)
   if isa(rattle, Bool)
      warning("Deprecation: Testing.fdtest: rattle should be a float")
      rattle ? rattle = 0.01 : rattle = 0.0
   end

   X0 = copy(positions(at))
   calc0 = calculator(at)
   cons0 = constraint(at)
   # if no constraint is attached, then attach the NullConstraint
   if typeof(constraint(at)) == NullConstraint
      set_constraint!(at, FixedCell(at))
   end
   set_calculator!(at, calc)
   # random perturbation to positions (and cell for VariableCell)
   # perturb atom positions a bit to get out of equilibrium states
   # Don't use `rattle!` here which screws up the `VariableCell` constraint
   # test!!! (but why?!?!?)
   x = dofs(at)
   x += rattle * rand(length(x))
   # call the actual FD test
   fdtest( x -> energy(at, x),
           x -> gradient(at, x),
           x )
   # restore original atom positions
   set_positions!(at, X0)
   set_calculator!(at, calc0)
   set_constraint!(at, cons0)
end


end
