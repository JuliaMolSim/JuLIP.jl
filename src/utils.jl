
export rattle!, r_sum, r_dot


############################################################
###   Some useful utility functions
############################################################
###   Robust Summation and Dot Products
############################################################

"Robust summation. Uses `sum_kbn`."
r_sum(a) = sum_kbn(a)


## NOTE: if I see this correctly, then r_dot allocates a temporary
##       vector, which is likely quite a performance overhead.
##       probably, we want to re-implement this.
## TODO: new version without the intermediate allocation
"Robust inner product. Defined as `r_dot(a, b) = r_sum(a .* b)`"
r_dot(a, b) = r_sum(a[:] .* b[:])


"""
`rattle!(at::AbstractAtoms, r::Float64; rnn = 1.0, respect_constraint = true)
  -> at`

randomly perturbs the atom positions

* `r`: magnitude of perturbation
* `rnn` : nearest-neighbour distance
* `respect_constraint`: set false to also perturb the constrained atom positions
"""
function rattle!(at::AbstractAtoms, r::Float64; rnn = 1.0, respect_constraint = true)
   # if there is no constraint, then revert to respect_constraint = false
   if isa(constraint(at), NullConstraint)
      respect_constraint = false
   end
   if respect_constraint
      x = dofs(at)
      x += r * rnn * 2.0/sqrt(3) * (rand(length(x)) - 0.5)
      set_dofs!(at, x)
   else
      X = positions(at) |> mat
      X += r * rnn * 2.0/sqrt(3) * (rand(size(X)) - 0.5)
      set_positions!(at, X)
   end
   return at
end


"""
use this instead of `warn`, then warnings can be turned off by setting
`Main.JULIPWARN=false`
"""
function julipwarn(s)
   if isdefined(Main, :JULIPWARN)
      if Main.JULIPWARN == false
         return false
      end
   end
   warn(s)
end
