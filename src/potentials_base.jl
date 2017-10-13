
export @D, @DD, @GRAD, @pot



# ===========================================================================
#     implement some fun little macros for easier access
#     to the potentials
# ===========================================================================

# Julia 0.4 version:
# call(pp::Potential, varargs...) = evaluate(pp, varargs...)
# call(pp::Potential, ::Type{Val{:D}}, varargs...) = evaluate_d(pp, varargs...)
# call(pp::Potential, ::Type{Val{:DD}}, varargs...) = evaluate_dd(pp, varargs...)
# call(pp::Potential, ::Type{Val{:GRAD}}, varargs...) = grad(pp, varargs...)

# unfortunately, in 0.5 `call` doesn't take an abstract argument anymore,
# which means that we need to specify for every potential how to
# create this syntactic sugar. This is what `@pot` is for.

"""
Annotate a type with `@pot` to setup the syntax sugar
for `evaluate, evaluate_d, evaluate_dd, grad`.

## Usage:

For example, the declaration
```julia
@pot type LennardJones <: PairPotential
   r0::Float64
end
"documentation for `LennardJones`"
LennardJones
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
   @assert fsig.head == :type
   tname, tparams = t_info(fsig.args[2])
   # isa(tname, Symbol) ? name_only = tname : name_only = tname.args[1]
   # @show name_only
   tname = esc(tname)
   for n = 1:length(tparams)
      tparams[n] = esc(tparams[n])
   end
   sym = esc(:x)
   quote
      $(esc(fsig))
      # Docs.@__doc__ $(name_only)
      ($sym::$tname){$(tparams...)}(args...) = evaluate($sym, args...)
      ($sym::$tname){$(tparams...)}(::Type{Val{:D}}, args...) = evaluate_d($sym, args...)
      ($sym::$tname){$(tparams...)}(::Type{Val{:DD}}, args...) = evaluate_dd($sym, args...)
      ($sym::$tname){$(tparams...)}(::Type{Val{:GRAD}}, args...) = grad($sym, args...)
   end
end

# t_info extracts type name as symbol and type parameters as an array
t_info(ex::Symbol) = (ex, tuple())
t_info(ex::Expr) = ex.head == :(<:) ? t_info(ex.args[1]) : (ex, ex.args[2:end])

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



# ==================================
#   Basic Potential Arithmetic

@pot type SumPot{P1, P2} <: PairPotential
   p1::P1
   p2::P2
end

# "sum of two pair potentials"
# SumPot

import Base.+
+(p1::PairPotential, p2::PairPotential) = SumPot(p1, p2)
evaluate(p::SumPot, r) = p.p1(r) + p.p2(r)
evaluate_d(p::SumPot, r) = (@D p.p1(r)) + (@D p.p2(r))
cutoff(p::SumPot) = max(cutoff(p.p1), cutoff(p.p2))
function Base.show(io::Base.IO, p::SumPot)
   print(io, p.p1)
   print(io, " + ")
   print(io, p.p2)
end

@pot type ProdPot{P1, P2} <: PairPotential
   p1::P1
   p2::P2
end
"product of two pair potentials"
ProdPot
import Base.*
*(p1::PairPotential, p2::PairPotential) = ProdPot(p1, p2)
evaluate(p::ProdPot, r) = p.p1(r) * p.p2(r)
evaluate_d(p::ProdPot, r) = (p.p1(r) * (@D p.p2(r)) + (@D p.p1(r)) * p.p2(r))
evaluate_dd(p::ProdPot, r) = (p.p1(r) * (@DD p.p2(r)) +
              2 * (@D p.p1(r)) * (@D p.p2(r)) + (@DD p.p1(r)) * p.p2(r))
cutoff(p::ProdPot) = min(cutoff(p.p1), cutoff(p.p2))
function Base.show(io::Base.IO, p::ProdPot)
   print(io, p.p1)
   print(io, " * ")
   print(io, p.p2)
end

# expand usage of prodpot to be useful for TightBinding.jl
# TODO: make sure this is consistent and expand this to other things!
#       basically, we want to allow that
#       a pair potential can depend on direction as well!
#       in this case, @D is already the gradient and @GRAD remaind undefined?
evaluate{P1,P2}(p::ProdPot{P1,P2}, r, R) = p.p1(r, R) * p.p2(r, R)
evaluate_d(p::ProdPot, r, R) = p.p1(r,R) * (@D p.p2(r,R)) + (@D p.p1(r,R)) * p.p2(r,R)
