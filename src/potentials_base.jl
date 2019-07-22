
export @D, @DD, @GRAD, @pot


# ===========================================================================
#     implement some fun little macros for easier access
#     to the potentials
# ===========================================================================

# REMARK on @pot:
# ---------------
# Julia 0.4 version:
# call(pp::Potential, varargs...) = evaluate(pp, varargs...)
# call(pp::Potential, ::Type{Val{:D}}, varargs...) = evaluate_d(pp, varargs...)
# call(pp::Potential, ::Type{Val{:DD}}, varargs...) = evaluate_dd(pp, varargs...)
# call(pp::Potential, ::Type{Val{:GRAD}}, varargs...) = grad(pp, varargs...)
# unfortunately, in 0.5 `call` doesn't take anabstractargument anymore,
# which means that we need to specify for every potential how to
# create this syntactic sugar. This is what `@pot` is for.

"""
Annotate a type with `@pot` to setup the syntax sugar
for `evaluate, evaluate_d, evaluate_dd, grad`.

## Usage:

For example, the declaration
```julia
"documentation for `LennardJones`"
mutable struct LennardJones <: PairPotential
   r0::Float64
end
@pot LennardJones
```
creates the following aliases:
```julia
lj = LennardJones(1.0)
lj(args...) = evaluate(lj, args...)
@D lj(args...) = evaluate_d(lj, args...)
@DD lj(args...) = evaluate_dd(lj, args...)
@GRAD lj(args...) = grad(lj, args...)
```

Usage of `@pot` is not restricted to pair potentials, but can be applied to
*any* type.
"""
macro pot(fsig)
   @assert fsig isa Symbol
   sym = esc(:x)
   tsym = esc(fsig)
   quote
      @inline ($sym::$tsym)(args...) = evaluate($sym, args...)
      @inline ($sym::$tsym)(::Type{Val{:D}}, args...) = evaluate_d($sym, args...)
      @inline ($sym::$tsym)(::Type{Val{:DD}}, args...) = evaluate_dd($sym, args...)
      @inline ($sym::$tsym)(::Type{Val{:GRAD}}, args...) = grad($sym, args...)
   end
end


# --------------------------------------------------------------------------

# next create macros that translate
"""
`@D`: Use to evaluate the derivative of a potential. E.g., to compute the
Lennard-Jones potential,
```julia
lj = LennardJones()
r = 1.0 + rand(10)
ϕ = lj(r)
ϕ' = @D lj(r)
```
see also `@DD`.
"""
macro D(fsig::Expr)
    @assert fsig.head == :call
    insert!(fsig.args, 2, Val{:D})
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    return fsig
end

"`@DD` : analogous to `@D`"
macro DD(fsig::Expr)
    @assert fsig.head == :call
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    insert!(fsig.args, 2, Val{:DD})
    return fsig
end

"`@GRAD` : analogous to `@D`, but escapes to `grad`"
macro GRAD(fsig::Expr)
    @assert fsig.head == :call
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    insert!(fsig.args, 2, Val{:GRAD})
    return fsig
end
