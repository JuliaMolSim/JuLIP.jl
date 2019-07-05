
using Parameters
import Base.Dict

export Atoms

"""
`JData`: datatype for storing any data

some data which needs to be updated if the configuration (positions only!) has
changed too much.
"""
mutable struct JData
   max_change::Float64     # how much X may change before recomputing
   accum_change::Float64   # how much has it changed already
   data::Any
end


"""
`Atoms{T <: AbstractFloat} <: AbstractAtoms`

The main type storing information about atomic configurations. This is
not normally constructed directly by calling `Atoms`, but using one of the
utility functions such as `bulk`.

## Constructors

TODO

## Convenience functions

* `bulk`

# Getters and Setters

* `positions`, `set_positions!`
* `momenta`, `set_momenta!`
* `masses`, `set_masses!`
* `numbers`, `set_numbers!`
* TODO: `chemical_symbols`, `set_chemical_symbols!`
* `cell`, `set_cell!`
* `pbc`, `set_pbc!`
* `calculator`, `set_calculator!`
* `constraint`, `set_constraint!`
* `get_data`, `set_data!`
"""
@with_kw mutable struct Atoms{T <: AbstractFloat} <: AbstractAtoms{T}
   X::Vector{JVec{T}} = JVec{T}[]       # positions
   P::Vector{JVec{T}} = JVec{T}[]       # momenta (or velocities?)
   M::Vector{T} = T[]                   # masses
   Z::Vector{Int16} = Int16[]                      # atomic numbers
   cell::JMat{T} = zero(JMat{T})                   # cell
   pbc::JVec{Bool} = JVec(false, false, false)     # boundary condition
   calc::Union{Nothing, AbstractCalculator} = nothing
   cons::Union{Nothing, AbstractConstraint} = nothing
   data::Dict{Any,JData} = Dict{Any,JData}()
end

Base.eltype(::Atoms{T}) where {T} = T

# default type is Float64
Atoms(;kwargs...) = Atoms{Float64}(;kwargs...)

_auto_pbc(pbc::Tuple{Bool, Bool, Bool}) = pbc
_auto_pbc(pbc::Bool) = (pbc, pbc, pbc)
_auto_pbc(pbc::AbstractVector) = tuple(pbc...)

_auto_cell(cell) = cell
_auto_cell(cell::AbstractMatrix) = JMat(cell)
_auto_cell(C::AbstractVector{T}) where {T <: Real} = (
   length(C) == 3 ? JMatF(C[1], 0.0, 0.0, 0.0, C[2], 0.0, 0.0, 0.0, C[3]) :
                    JMatF(C...) )  # (case 2 requires that length(C) == 0
_auto_cell(C::AbstractVector{T}) where {T <: AbstractVector} = JMatF([ C[1] C[2] C[3] ])
_auto_cell(C::AbstractVector) = _auto_cell([ c for c in C ])

# if we have no clue about X just return it and see what happens
_auto_X(X) = X
# if the elements of X weren't inferred, try to infer before reading
# TODO: this might lead to a stack overflow!!
_auto_X(X::AbstractVector) = _auto_X( [x for x in X] )
# the cases where we know what to do ...
_auto_X(X::AbstractVector{T}) where {T <: AbstractVector} = [ JVecF(x) for x in X ]
_auto_X(X::AbstractVector{T}) where {T <: Real} = _auto_X(reshape(X, 3, :))
_auto_X(X::AbstractMatrix) = (@assert size(X)[1] == 3;
                              [ JVecF(X[:,n]) for n = 1:size(X,2) ])

_auto_M(M::AbstractVector) = Vector{Float64}(M)

_auto_Z(Z::AbstractVector) = _auto_Z([z for z in Z])
_auto_Z(Z::Vector{TI}) where {TI <: Integer}  =
            isconcretetype(TI) ? Z : Vector{Int}(Z)



Atoms(X, P, M, Z, cell, pbc;
      calc = nothing,
      cons = nothing,
      data = Dict{Any,JData}()) =
   Atoms(_auto_X(X),
         _auto_X(P),
         _auto_M(M),
         _auto_Z(Z),
         _auto_cell(cell),
         _auto_pbc(pbc),
         calc,
         cons,
         data)

Atoms(Z::Vector{TI}, X::Vector{JVec{T}}; kwargs...) where {TI, T} =
  Atoms{T}(;Z=Z, X=X, P = zeros(eltype(X), length(X)), M = zeros(length(X)), kwargs...)

# derived properties
length(at::Atoms) = length(at.X)

getindex(at::Atoms, i::Integer) = at.X[i]
function setindex!(at::Atoms{T}, val, i::Integer) where {T}
   at.X[i] = convert(JVec{T}, val)
   return val
end

# ------------------------ access to struct fields ----------------------

# ----- getters

positions(at::Atoms) = copy(at.X)
momenta(at::Atoms) = copy(at.P)
masses(at::Atoms) = copy(at.M)
numbers(at::Atoms) = copy(at.Z)
atomic_numbers(at::Atoms) = copy(at.Z)
cell(at::Atoms) = at.cell
pbc(at::Atoms) = at.pbc
calculator(at::Atoms) = at.calc
constraint(at::Atoms) = at.cons

chemical_symbols(at::Atoms) = Chemistry.chemical_symbol.(at.Z)

# ----- setters

function set_positions!(at::Atoms{T}, X::AbstractVector{JVec{T}})  where T
   update_data!(at, dist(at, X))
   at.X .= X
   return at
end
function set_momenta!(at::Atoms{T}, P::AbstractVector{JVec{T}})  where T
   at.P .= P
   return at
end
function set_masses!(at::Atoms{T}, M::AbstractVector{T})  where T
   update_data!(at, Inf)
   at.M .= M
   return at
end
function set_numbers!(at::Atoms, Z::AbstractVector{<:Integer})
   update_data!(at, Inf)
   at.Z .= Z
   return at
end
function set_cell!(at::Atoms{T}, C::AbstractMatrix) where T
   update_data!(at, Inf)
   at.cell = JMat{T}(C)
   return at
end
function set_pbc!(at::Atoms, p::Union{AbstractVector, Tuple})
   q = JVec{Bool}(p...)
   if at.pbc != q
      update_data!(at, Inf)
      at.pbc = q
   end
   return at
end
function set_calculator!(at::Atoms, calc::AbstractCalculator)
   at.calc = calc
   return at
end
function set_constraint!(at::Atoms, cons::AbstractConstraint)
   at.cons = cons
   return at
end

# --------------- equality tests -------------------

import Base.==
==(at1::Atoms{T}, at2::Atoms{T}) where T = (
   isapprox(at1, at2, tol = 0.0)  &&
          (at1.data == at2.data)  &&
          (at1.calc == at2.calc)  &&
          (at1.cons == at2.cons) )

import Base.isapprox
function isapprox(at1::Atoms{T}, at2::Atoms{T}; tol = sqrt(eps(T))) where {T}
   if length(at1) != length(at2)
      return false
   end
   if tol > 0
      ndigits = floor(Int, abs(log10(tol)))
      X1 = [round.(x, ndigits) for x in at1.X]
      X2 = [round.(x, ndigits) for x in at2.X]
   else
      X1, X2 = at1.X, at2.X
   end
   p1 = sortperm(X1)
   p2 = sortperm(X2)
   return (maxdist(at1.X[p1], at2.X[p2]) <= tol) &&
          (maxdist(at1.P[p1], at2.P[p2]) <= tol)  &&
          (norm(at1.M[p1] - at2.M[p2], Inf) <= tol) &&
          (at1.Z[p1] == at2.Z[p2]) &&
          (norm(at1.cell - at2.cell, Inf) <= tol) &&
          (at1.pbc == at2.pbc)
end

# --------------- some aliases for easier / direct access --------------
#                 and some access not covered above


function neighbourlist(at::Atoms, cutoff::AbstractFloat; recompute=false, kwargs...)
   # todo -> re-design this from scratch . . .
   PairList(positions(at), cutoff, cell(at), pbc(at); kwargs...)
   # if !has_data(at, (:nlist, cutoff)) || recompute
   #    set_transient!(at, (:nlist, cutoff),
   #          PairList(positions(at), cutoff, cell(at), pbc(at))
   #       )
   # end
   # return get_data(at, (:nlist, cutoff))
end

neighbourlist(at::Atoms) = neighbourlist(at, cutoff(at))


# --------------- data Dict handling ------------------

"""
`has_data(at::Atoms, key)`: checks whether `at` is storing a datum with key `key`; see also `set_data!`
"""
has_data(at::Atoms, key) = haskey(at.data, key)

"""
`get_data(at::Atoms, key)`: retrieves a datum;
see also `set_data!`. Unlike `get_positions`, etc, this returns the original
datum, not a copy (unless it is immutable).
"""
get_data(at::Atoms, key) = (at.data[key]).data

"""
`set_data!(a::Atoms, key, value, max_change=Inf)`:

Stores `value` under an arbitrary key `key` inside `at`. The entry is paired
with an updatemeasure `chg`. Whenever atom positions or the cell change by a
distance `d`, the counter is incremented `chg += d`. When `chg > max_change`,
the entry is deleted from the dictionary.

The two most common uses are with `max_change = Inf` (default) - in this case
the entry will never be deleted; and `max_change = 0.0` - in this case the entry
will be deleted whenever atom positions or the cell change.

The change counter is used, e.g., for neighbourlists with buffer, and for
preconditioners so that the list need not be recomputed each time the
configuration changes.
"""
function set_data!(at::Atoms, key, value, max_change=Inf)
   at.data[key] = JData(max_change, 0.0, value)
   return at
end

"""
`function set_transient!(a::Atoms, key, value, max_change=0.0)`: this is an
alias for `set_data!` but with default `max_change = 0.0` instead of `max_change =
Inf`.
"""
set_transient!(at::Atoms, key, value, max_change=0.0) =
      set_data!(at, key, value, max_change)

"""
`update_data!(at::Atoms, r::Real)`:
increment the change counter in all stored data and delete the
expired ones.
"""
function update_data!(at::Atoms, r::Real)
   for (key, t) in at.data
      t.accum_change += r
      if t.accum_change >= t.max_change
         delete!(at.data, key)
      end
   end
   return at
end



# ------------------------ FIO  ----------------------



Dict(at::Atoms) =
   Dict( "__id__" => "JuLIP_Atoms",
         "X"      => at.X,
         "P"      => at.P,
         "M"      => at.M,
         "Z"      => at.Z,
         "cell"   => at.cell,
         "pbc"    => at.pbc,
         "calc"   => nothing,
         "cons"   => nothing,
         "data"   => nothing )

Atoms(D::Dict) = Atoms(D["X"], D["P"], D["M"], D["Z"], D["cell"], D["pbc"])
          # calc = D["calc"],
          # cons = D["cons"],
          # data = D["data"] )

Base.convert(::Val{:JuLIP_Atoms}, D) = Atoms(D)
