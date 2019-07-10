
## Non-allocating calculations

This document discusses the implementation details of the (potentially)
non-allocating functions `energy!, forces!, virial!, evaluate!, evaluate_d!`.
JuLIP implements a fall-back to the standard functions `energy, forces, ...`
but a calculator that requires large allocations should implement non-allocating
versions in order to give the option of higher-performance evaluations. This
is especially useful for regression where thousands of evaluations are
required.

In a second step this framework should be rewritten so that the non-allocating
verions are implemented by all calculators while the allocating versions
are just wrappers.

The non-allocating calling conventions are
```{julia}
energy!(   tmp,  calc, at)
forces!(F, tmpd, calc, at)
virial!(   tmp,  calc, at)
```
where
```{julia}
tmp = alloc_temp(  calc, at)
tmpd = alloc_temp_d(calc, at)
```

For site energies the conventions are
```{julia}
evaluate!(tmp, calc, args...)
evaluate_d!(out, tmpd, calc, args...)
```
where
```{julia}
tmp = alloc_temp(calc, N)
tmpd = alloc_temp_d(calc, N)
```
and `N` is an integer giving an upper bound on the number of neighbours
in the system.
