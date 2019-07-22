
## Calculator Implementations

The most important user-oriented functions are
```{julia}
energy(calc, at)
forces(calc, at)
virial(calc, at)
```

A new calculators / model can either directly implement these, or alternatively
one of the functions that are called in their prototype implementation.
Concretely, if `calc` is an interatomic potential with a site energy, then
these outer calculators should *not* be implemented, but rather the evaluation
of the site energy and its derivatives. But e.g. for a Tight-binding model
or an interface to another library, one could or possibly must directly
implement `energy, force, virial`.

The standard implementation of `energy, forces, virial` is designed to
admit minimal or even zero allocations during these evaluations. This is
achieved as follows: first `energy, forces, virial` call in-place
versions with the calling convention
```{julia}
energy!(   tmp,  calc, at)
forces!(F, tmpd, calc, at)
virial!(   tmp,  calc, at)
```
where
```{julia}
tmp = alloc_temp(   calc, at)
tmpd = alloc_temp_d(calc, at)
```
A new calculator could again implement `energy!, forces!, virial!`
directly.

### Site Potentials

However, for a site potentials one should require that
```
calc <: SitePotential
```
In that case an implementation of `energy!, forces!, virial!`
already exists, which is based on
```{julia}
evaluate!(      tmp,  calc, R)
evaluate_d!(dV, tmpd, calc, R)
```
where `R::AbstractVector{<:JVec}`.

Now, the temporary arrays are obtained from
```{julia}
tmp = alloc_temp(calc, N)
tmpd = alloc_temp_d(calc, N)
```
and `N` is an integer giving an upper bound on the number of neighbours
in the system. The allocating versions `energy, ...` will first generate
a neighbourlist, then calculate `N` (cf. `NeighbourLists.maxneigs`)
then allocate `tmp, tmpd` and then use that to call the non-allocating
versions `energy!, ...`.

It is *required* that `tmp` has a field `temp.R` and that
`tmpd` has fields `tmpd.dV, tmp.R` which will be used to
store the neighbourhood information and the site energy derivatives.

### Pair Potentials

For pair potentials, i.e. `calc <: PairPotential <: SitePotential`
the functions
```{julia}
evaluate!(       tmp, calc, R::AbstractVector{<:JVec})
evaluate_d!(dV, tmpd, calc, R::AbstractVector{<:JVec})
```
are implemented through calls to only
```{julia}
evaluate!(   tmp, calc, r::Number)
evaluate_d!(tmpd, calc, r::Number)
```
which in turn are implemented as
```
evaluate(calc, r)
evaluate_d(calc, r)
```
A new implementation of a `PairPotential` may overload these definitions
at any of these three levels of implementation.

### Hessians and Preconditioners

The hessian implementation is less concerned about memory management. The
thought is that once we are using hessians all hopes to keep memory under
control is out the window anyhow. Therefore, there is only one hessian
implementation - for `SitePotential`s at least - based on
```{julia}
hessian_pos(calc, at)
```
which allocates temporary arrays for `calc` via
```{julia}
tmpdd = alloc_temp_dd(calc, N)
```
where `N` is again the maximum number of neighbours. It also allocates a
temporary storage `hEs` for the hessian blocks. (This does *not* need to be
in `tmpdd` now!) Then the hessian assembly will call
```{julia}
evaluate_dd(hEs, tmp, calc, R::AbstractVector{<:JVec})
```

For preconditioners the framework is the same but `evaluate_dd` is
replaced by `precon`.
