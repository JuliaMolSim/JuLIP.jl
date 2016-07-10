
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
