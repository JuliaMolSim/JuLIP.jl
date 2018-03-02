
# TODO:
#       defm, set_defm!   >>> decide what to do about those
#       Base.read, Base.write   >>> rename as read_xyz, write_xyz



using Parameters

export Atoms,
       bulk,
       cell_vecs



"""
`JData`: datatype for storing any data

some data which needs to be updated if the configuration (positions only!) has
changed too much.
"""
mutable struct JData{T <: AbstractFloat}
   max_change::T     # how much X may change before recomputing
   accum_change::T   # how much has it changed already
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
   X::JVecs{T} = JVecs{T}[]       # positions
   P::JVecs{T} = JVecs{T}[]       # momenta (or velocities?)
   M::Vector{T} = T[]             # masses
   Z::Vector{TI} = TI[]           # atomic numbers
   cell::JMat{T} = zero(JMat{T})                   # cell
   pbc::JVec{Bool} = JVec(false, false, false)     # boundary condition
   calc::AbstractCalculator = NullCalculator()
   cons::AbstractConstraint = NullConstraint()
   data::Dict{Any,JData{T}} = Dict{Any,JData{T}}()
end

Atoms() = Atoms{Float64}()

# derived properties
length(at::Atoms) = length(at.X)

# ------------------------ access to struct fields ----------------------

# ----- getters

positions(at::Atoms) = copy(at.X)
momenta(at::Atoms) = copy(at.P)
masses(at::Atoms) = copy(at.M)
numbers(at::Atoms) = copy(at.Z)
cell(at::Atoms) = at.cell
pbc(at::Atoms) = at.pbc
calculator(at::Atoms) = at.calc
constraint(at::Atoms) = at.cons

# ----- setters

function set_positions!(at::Atoms{T}, X::Vector{JVec{T}})  where T
   update_data!(at, dist(at, X))
   at.X .= X
   return at
end
function set_momenta!(at::Atoms{T}, P::Vector{JVec{T}})  where T
   at.P .= P
   return at
end
function set_masses!(at::Atoms{T}, X::Vector{T})  where T
   update_data!(at, Inf)
   at.M .= M
   return at
end
function set_numbers!(at::Atoms{T, TI}, Z::Vector{TI}) where T where TI
   update_data!(at, Inf)
   at.Z .= Z
   return at
end
function set_cell!(at::Atoms{T}, C::AbstractMatrix) where T
   at.cell = JMat{T}(C)
   return at
end
function set_pbc(at::Atoms, p)
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


# --------------- some aliases for easier / direct access --------------
#                 and some access not covered above

# access positions by direct indexing
Base.getindex(at::Atoms, i::Integer) = at.X[i]
function Base.setindex!(at::Atoms{T}, i::Integer, x::JVec) where T <: AbstractFloat
   at.X[i] = JVec{T}(x)
   return at.X[i]
end

"""
`cell_vecs(at::Atoms)` : return the three cell vectors
"""
cell_vecs(at::Atoms) = at.cell[1,:], at.cell[2,:], at.cell[3,:]



function neighbourlist(at::Atoms{T}, cutoff::T; recompute=false) where T
   # TODO: re-design this from scratch . . .
   if !has_data(at, (:nlist, cutoff)) || recompute
      set_transient!(at, (:nlist, cutoff),
            CellList(positions(at), cutoff, cell(at), pbc(at))
         )
   end
   return get_data(at, (:nlist, cutoff))
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
            CellList(positions(at), cutoff, cell(at), pbc(at))
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
get_data(at::Atoms, key) = (a.data[key]).data

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
   at.transient[name] = JData(max_change, 0.0, value)
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
   for (key, t) in a.data
      t.accum_change += r
      if t.accum_change >= t.max_change
         delete!(at.data, key)
      end
   end
   return at
end




# ------------------ Some Basic ASE Functionality Reproduced -----------------


"""
`deleteat!(at::Atoms, n) -> at`:

returns the same atoms object `at`, but with the atom(s) specified by `n`
removed.
"""
function Base.deleteat!(at::Atoms, n)
   deleteat!(at.X, n)
   deleteat!(at.P, n)
   deleteat!(at.M, n)
   deleteat!(at.Z, n)
   update_data!(at, Inf)
   return at
end

"""
```
repeat(at::Atoms, n::NTuple{3}) -> Atoms
repeat(at::Atoms, n::Integer) = repeat(at, (n,n,n))
```

Takes an `Atoms` configuration / cell and repeats it n_j times
into the j-th cell-vector direction. For example,
```
at = repeat(bulk("C"), (3,2,4))
```
creates 3 x 2 x 4 unit cells of carbon.

The same can be achieved by `*`:
```
at = bulk("C") * (3, 2, 4)
```
"""
function repeat(at::Atoms, n::NTuple{3})
   C = cell(at)
   c1, c2, c3 = cell_vecs(at)

end

repeat(at::Atoms, n::Integer) = repeat(at, (n,n,n))

Base.*(at::Atoms, n) = repeat(at, n)
Base.*(n, at::Atoms) = repeat(at, n)


# ------------------ Conversion from ASE Objects -----------------

Atoms(at_ase::ASE.ASEAtoms) =
   Atoms( positions(at_ase),
          momenta(at_ase),
          ASE.masses(at_ase),
          Int[],
          [Symbol(s) for s in ASE.chemical_symbols(at_ase)],
          JMat{Float64}(cell(at_ase)),
          pbc(at_ase),
          calculator(at_ase),
          constraint(at_ase),
          Dict{Any,JData{Float64}}() )

bulk(s::Symbol, pbc = (true, true, true), cubic=false, repeat = (1,1,1)) =
      Atoms(set_pbc!(ASE.bulk(string(s), cubic=cubic), pbc) * repeat)
