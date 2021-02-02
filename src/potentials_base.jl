
import JuLIP: read_dict, write_dict
export @D, @DD, @GRAD, @pot

using JuLIP: AtomicNumber


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


# ----------------------------------------------------------------------
#    Managing a list of species
# ----------------------------------------------------------------------

abstract type AbstractZList end

"""
`ZList` and `SZList{NZ}` : simple data structures that store a list
of species and convert between atomic numbers and the index in the list.
Can be constructed via
* `ZList(zors)` : where `zors` is an Integer  or `Symbol` (single species)
* `ZList(zs1, zs2, ..., zsn)`
* `ZList([sz1, zs2, ..., zsn])`
* All of these take a kwarg `static = {true, false}`; if `true`, then `ZList`
will return a `SZList{NZ}` for (possibly) faster access.
"""
struct ZList <: AbstractZList
   list::Vector{AtomicNumber}
end

Base.length(zlist::AbstractZList) = length(zlist.list)


struct SZList{N} <: AbstractZList
   list::SVector{N, AtomicNumber}
end

function ZList(zlist::AbstractVector{<: Number};
               static = false, sorted=true)
   sortfun = sorted ? sort : identity
   return (static ? SZList(SVector( (AtomicNumber.(sortfun(zlist)))... ))
                  :  ZList( convert(Vector{AtomicNumber}, sortfun(zlist)) ))
end

ZList(s::Symbol; kwargs...) =
      ZList( [ atomic_number(s) ]; kwargs... )

ZList(S::AbstractVector{Symbol}; kwargs...) =
      ZList( atomic_number.(S); kwargs... )

ZList(args...; kwargs... ) =
      ZList( [args...]; kwargs...)


i2z(Zs::AbstractZList, i::Integer) = Zs.list[i]

function z2i(Zs::AbstractZList, z::AtomicNumber)
   if Zs.list[1] == JuLIP.Chemistry.__zAny__
      return 1
   end
   for j = 1:length(Zs.list)
      if Zs.list[j] == z
         return j
      end
   end
   error("z = $z not found in ZList $(Zs.list)")
end

zlist(V) = V.zlist
i2z(V, i::Integer) = i2z(zlist(V), i)
z2i(V, z::AtomicNumber ) = z2i(zlist(V), z)
numz(V) = length(zlist(V))

write_dict(zlist::ZList) = Dict("__id__" => "JuLIP_ZList",
                                  "list" => Int.(zlist.list))

read_dict(::Val{:JuLIP_ZList}, D::Dict) = ZList(D)
ZList(D::Dict) = ZList(D["list"])

write_dict(zlist::SZList) = Dict("__id__" => "JuLIP_SZList",
                                 "list" => Int.(zlist.list))
read_dict(::Val{:JuLIP_SZList}, D::Dict) = SZList(D)
SZList(D::Dict) = ZList([D["list"]...], static = true)
