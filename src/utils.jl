
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
`rattle!(at::AbstractAtoms, r::Float64; rnn = 1.0)
  -> at`

randomly perturbs the atom positions

* `r`: magnitude of perturbation
* `rnn` : nearest-neighbour distance
"""
function rattle!(at::AbstractAtoms, r::Float64; rnn = 1.0)
   X = positions(at) |> mat
   X += r * rnn * 2.0/sqrt(3) * (rand(size(X)) - 0.5)
   return set_positions!(at, X)
end
