# TODO:
# * defm, set_defm!         >>> decide what to do about those
# * Base.read, Base.write   >>> rename as read_xyz, write_xyz
# * if we make Atoms immutable, then we could make the calculator
#   etc a type parameter, but it has the disadvantage that we would need
#   to create a new object every time we change it significantly.
#   maybe a good thing.
#
# * `JData` has a hard-coded floating point type. this seems necessary
#       because otherwise the kw-constructor doesn't work. probably this is
#       not important, but could be fixed at some point (low priority)
#
# * the kwconstructor only allows Float64, this is a bad shortcoming that
#   needs to be fixed I think
#
# * decide whether == should check equality of data
#
# * implement a softer `isequal` which checks for "physical" equality, i.e.,
#   it accounts for PBC, ignores any data, and allows for a controllable
#   error in the positions, momenta and masses
#
# * add fields for charges and magnetic moments?
#

const CH = JuLIP.Chemistry

import Base.Dict

export Atoms,
       bulk,
       cell_vecs,
       chemical_symbols,
       atomic_numbers, numbers,
       chemical_symbols


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
@with_kw mutable struct Atoms{T <: AbstractFloat, TI <: Integer} <: AbstractAtoms
   X::Vector{JVec{T}} = JVec{T}[]       # positions
   P::Vector{JVec{T}} = JVec{T}[]       # momenta (or velocities?)
   M::Vector{T} = T[]             # masses
   Z::Vector{TI} = Int[]           # atomic numbers
   cell::JMat{T} = zero(JMat{T})                   # cell
   pbc::JVec{Bool} = JVec(false, false, false)     # boundary condition
   calc::Union{Nothing, AbstractCalculator} = nothing
   cons::Union{Nothing, AbstractConstraint} = nothing
   data::Dict{Any,JData} = Dict{Any,JData}()
end

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
_auto_X(X::AbstractVector{T}) where {T <: Real} = _auto_V(reshape(X, 3, :))
_auto_X(X::AbstractMatrix) = (@assert size(X)[1] == 3;
                              [ JVecF(X[:,n]) for n = 1:size(X,2) ])

_auto_M(M::AbstractVector) = Vector{Float64}(M)

_auto_Z(Z::AbstractVector) = _auto_Z([z for z in Z])
_auto_Z(Z::Vector{TI}) where {TI <: Integer}  =
            isconcretetype(TI) ? Z : Vector{Int}(Z)



Atoms(X, P, M, Z, cell, pbc; calc=NullCalculator(),
      cons = NullConstraint(), data = Dict{Any,JData}()) =
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
  Atoms{T, TI}(;Z=Z, X=X, P = zeros(eltype(X), length(X)), M = zeros(length(X)), kwargs...)

# derived properties
length(at::Atoms) = length(at.X)

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
function set_numbers!(at::Atoms{T, TI}, Z::AbstractVector{TI}) where {T, TI}
   update_data!(at, Inf)
   at.Z .= Z
   return at
end
function set_cell!(at::Atoms{T}, C::AbstractMatrix) where T
   at.cell = JMat{T}(C)
   return at
end
function set_pbc!(at::Atoms, p::Union{AbstractVector, Tuple})
   at.pbc = JVec{Bool}(p...)
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

# we need to sort vectors of vectors to test equality of configurations
# note this is a strict notion of equality
function Base.isless(x::JVec, y::JVec)
   if x[1] < y[1]
      return true
   elseif x[1] == y[1]
      if x[2] < y[2]
         return true
      elseif x[2] == y[2]
         if x[3] < y[3]
            return true
         end
      end
   end
   return false
end

import Base.==
==(at1::Atoms{T,TI}, at2::Atoms{T,TI}) where T where TI =
   isapprox(at1, at2, tol = 0.0) && (at1.data == at2.data)

import Base.isapprox
function isapprox(at1::Atoms{T,TI}, at2::Atoms{T,TI}; tol = sqrt(eps(T))) where T where TI
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
          (maxdist(at1.P[p1], at2.P[p2]) <= tol) # &&
          (norm(at1.M[p1] - at2.M[p2], Inf) <= tol) &&
          (at1.Z[p1] == at2.Z[p2]) &&
          (norm(at1.cell - at2.cell, Inf) <= tol) &&
          (at1.pbc == at2.pbc) &&
          (at1.calc == at2.calc) &&
          (at1.cons == at2.cons)
end

# --------------- some aliases for easier / direct access --------------
#                 and some access not covered above

# access positions by direct indexing
Base.getindex(at::Atoms, i::Integer) = at.X[i]
function Base.setindex!(at::Atoms{T}, x::JVec, i::Integer) where T <: AbstractFloat
   at.X[i] = JVec{T}(x)
   return at.X[i]
end

"""
`cell_vecs(at::Atoms)` : return the three cell vectors
"""
cell_vecs(at::Atoms) = at.cell[1,:], at.cell[2,:], at.cell[3,:]


function neighbourlist(at::Atoms{T}, cutoff::T; recompute=false, kwargs...) where T <: AbstractFloat
   # TODO: re-design this from scratch . . .
   PairList(positions(at), cutoff, cell(at), pbc(at); kwargs...)
   # if !has_data(at, (:nlist, cutoff)) || recompute
   #    set_transient!(at, (:nlist, cutoff),
   #          PairList(positions(at), cutoff, cell(at), pbc(at))
   #       )
   # end
   # return get_data(at, (:nlist, cutoff))
end

neighbourlist(at::Atoms) = neighbourlist(at, cutoff(at))

"""
`static_neighbourlist(at::Atoms, cutoff)`

This function first checks whether a static neighbourlist already exists
with cutoff `cutoff` and if it does then it returns the existing list.
If it does not, then it computes a new neighbour list with the current
configuration, stores it for later use and returns it.
"""
function static_neighbourlist(at::Atoms{T}, cutoff::T) where T
   if !has_data(at, (:snlist, cutoff))
      set_data!( at, (:snlist, cutoff),
            PairList(positions(at), cutoff, cell(at), pbc(at))
         )
   end
   return get_data(at, (:snlist, cutoff))
end

static_neighbourlist(at::Atoms) = static_neighbourlist(at, cutoff(at))




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



# ------------------------ workaround for JLD bugs  ----------------------



# for the time being we won't store calculators, constraints and data
# TODO: this should be implemented asap
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
