
import AtomsBase
import Base.Dict
using Unitful

export Atoms

"""
`JData`: datatype for storing any data

some data which needs to be updated if the configuration (positions only!) has
changed too much.
"""
mutable struct JData{T}
   max_change::T     # how much X may change before recomputing
   accum_change::T   # how much has it changed already
   data::Any
end

JData(data) = JData(Inf, 0.0, data) 

struct LinearConstraint{T}
   C::Matrix{T}
   b::Vector{T}
   desc::String   # description of the constraint
end

mutable struct DofManager{T}
   variablecell::Bool
   xfree::Vector{Int}
   lincons::Vector{LinearConstraint{T}}
   # ----
   X0::Vector{JVec{T}}
   F0::JMat{T}
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
* `cell`, `set_cell!`
* `pbc`, `set_pbc!`
* `calculator`, `set_calculator!`
* `get_data`, `set_data!`
"""
mutable struct Atoms{T <: AbstractFloat} <: AbstractAtoms{T}
   X::Vector{JVec{T}}              # positions
   P::Vector{JVec{T}}              # momenta (or velocities?)
   M::Vector{T}                    # masses
   Z::Vector{AtomicNumber}         # atomic numbers
   cell::JMat{T}                   # cell
   pbc::JVec{Bool}                 # boundary condition
   calc::Union{Nothing, AbstractCalculator}
   dofmgr::DofManager
   data::Dict{Any,JData{T}}
end

function Atoms(
      X::Vector{JVec{T}},
      P::Vector{JVec{T}},
      M::Vector{T},
      Z::Vector{AtomicNumber},
      cell::JMat{T},
      pbc::JVec{Bool},
      calc::Union{Nothing, AbstractCalculator} = nothing;
      data::Union{Nothing, Dict{Any,JData{T}}} = nothing
   ) where {T}
   tmp = something(data, Dict{Any,JData{T}}())
   Atoms(X, P, M, Z, cell, pbc, calc,
         DofManager(length(X), T), tmp)
end


function Atoms(sys::AtomsBase.AbstractSystem)
   X = [ ustrip.(u"Å", AtomsBase.position(sys,i) ) for i in 1:length(sys)  ]
   V = [ ustrip.(u"eV^0.5/u^0.5", AtomsBase.velocity(sys,i) ) for i in 1:length(sys)  ]
   M = [ ustrip(u"u", AtomsBase.atomic_mass(sys,i) ) for i in 1:length(sys) ]
   Z = [ (AtomicNumber ∘ AtomsBase.atomic_number)(sys,i) for i in 1:length(sys) ]
   cell = map( x -> ustrip.(u"Å", x), sys[:bounding_box])
   pbc = map( x -> x == AtomsBase.Periodic ? true : false , AtomsBase.boundary_conditions(sys))
   data = Dict{Any,JData{eltype(M)}}( String(key)=>JData(sys[key]) for key in keys(sys) 
      if !( key in (:bounding_box, :boundary_conditions) )
   )
   return JuLIP.Atoms(X, M .* V, M, Z, hcat(cell...), pbc; data=data)
end


function AtomsBase.FlexibleSystem(sys::Atoms)
   atoms = map( 1:length(sys)  ) do i
       s = Int(sys.Z[i])
       r = sys[i] * u"Å"
       m = sys.M[i] * u"u"
       v = (sys.P[i] ./ sys.M[i]) * sqrt(u"eV/u")
       AtomsBase.Atom(s, r, v; atomic_mass=m)
   end
   pbc = map( sys.pbc ) do a
       a ? AtomsBase.Periodic() : AtomsBase.DirichletZero()
   end
   cell = [ c * u"Å" for c in eachrow(sys.cell) ]
   data = Dict(
      Symbol(key)=>val.data  for (key,val) in sys.data 
   )
   return AtomsBase.FlexibleSystem(atoms, cell, pbc; data...)
end

Base.convert(::Type{Atoms}, a::AtomsBase.AbstractSystem) = Atoms(a)
Base.convert(::Type{AtomsBase.FlexibleSystem}, a::Atoms) = AtomsBase.FlexibleSystem(a)
Base.convert(::Type{AtomsBase.AbstractSystem}, a::Atoms) = AtomsBase.FlexibleSystem(a)

fltype(::Atoms{T}) where {T} = T

function Base.show(io::IO, a::Atoms)
   print(io, "JuLIP.Atoms with $(length(a)) atoms")
end

# derived properties
length(at::Atoms) = length(at.X)

getindex(at::Atoms, i::Integer) = at.X[i]

function setindex!(at::Atoms{T}, val, i::Integer) where {T}
   u = norm(at.X[i] - val)
   at.X[i] = convert(JVec{T}, val)
   update_data!(at, u)
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
   at.Z .= AtomicNumber.(Z)
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
function set_calculator!(at::Atoms, calc::Union{AbstractCalculator, Nothing})
   at.calc = calc
   return at
end

# --------------- equality tests -------------------

import Base.==
==(at1::Atoms{T}, at2::Atoms{T}) where T = (
   isapprox(at1, at2, tol = 0.0)  &&
          (at1.data == at2.data)  &&
          (at1.calc == at2.calc)  &&
          (at1.dofmgr == at2.dofmgr) )

import Base.isapprox
function isapprox(at1::Atoms{T}, at2::Atoms{T}; tol = sqrt(eps(T))) where {T}
   if length(at1) != length(at2)
      return false
   end
   if tol > 0
      ndigits = floor(Int, abs(log10(tol)))
      X1 = [round.(x; digits=ndigits) for x in at1.X]
      X2 = [round.(x; digits=ndigits) for x in at2.X]
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


function neighbourlist(at::Atoms{T}, rcut::AbstractFloat;
            recompute=false, key="nlist:default", storelist=true,
            int_type = Int, kwargs...) where {T}
   # we are allowed to use a stored list
   if !recompute
      if has_data(at, key)
         nlist = get_data(at, key)::PairList{T, int_type}
         if cutoff(nlist) >= rcut
            return nlist
         end
      end
   end
   # we are either forces to recompute, or no nlist exists or is not
   # suitable for the request.
   nlist = PairList(positions(at), rcut, cell(at), pbc(at);
                    int_type=int_type, kwargs...)::PairList{T, int_type}
   if storelist
      set_data!(at, key, nlist, 0.0)
   end
   return nlist
end

# function neighbourlist(at::Atoms{T}, rcut::AbstractFloat;
#             recompute=false, key="nlist:default", kwargs...) where {T}
#    return PairList(positions(at), rcut, cell(at), pbc(at); kwargs...)
# end


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



JuLIP.FIO.write_dict(at::Atoms) =
   Dict( "__id__" => "JuLIP_Atoms",
         "X"      => at.X,
         "P"      => at.P,
         "M"      => at.M,
         "Z"      => Int.(at.Z),
         "cell"   => at.cell,
         "pbc"    => at.pbc,
         "calc"   => nothing,
         "data"   => nothing )

Atoms(D::Dict) = Atoms(D["X"], D["P"], D["M"], AtomicNumber.(D["Z"]),
                       D["cell"], D["pbc"])
          # calc = D["calc"],
          # cons = D["cons"],
          # data = D["data"] )

JuLIP.FIO.read_dict(::Val{:JuLIP_Atoms}, D::Dict) = Atoms(D)

import ExtXYZ

# conversion rules from Atoms.data entries to ExtXYZ standard types
_write_convert(value) = nothing # default rule: do not write
_write_convert(value::AbstractVector) = value
_write_convert(value::AbstractVector{JVec{T}}) where T = value |> mat |> Matrix
_write_convert(value::JVec{T}) where T = value |> Array
_write_convert(value::JMat{T}) where T = value |> Matrix
_write_convert(value::Union{Number,String}) = value

# (value, natoms) -> destination ∈ (:info, :arrays, missing)
_dest(::Any, natoms::Int) = missing

# this is a little fragile, since it assumes all vectors of length `natoms` contain per-atom data
# this is problematic for e.g. `natoms=3`.
_dest(value::AbstractVector, natoms::Int) = length(value) == natoms ? :arrays : :info
_dest(::JVec{T}, natoms::Int) where T = :info
_dest(::JMat{T}, natoms::Int) where T = :info
_dest(::Union{Number,String}, natoms::Int) = :info

function _atoms_to_extxyz_dict(atoms::Atoms{T}) where {T}
   dict = write_dict(atoms)
   natoms = length(atoms)
   dict["N_atoms"] = natoms
   dict["cell"] = reshape(_write_convert(dict["cell"]), 3, 3) #issue #149
   dict["pbc"] = _write_convert(dict["pbc"])

   # try to figure out whether data entries are per-atom or per-config
   info = Dict{String}{Any}()
   arrays = Dict{String}{Any}()
   for key in keys(atoms.data)
      value = get_data(atoms, key)
      dest, value = _dest(value, natoms), _write_convert(value)
      if value === nothing || ismissing(dest)
         continue
      end
      key = string(key) # symbol -> string, if needed
      if dest == :arrays
         push!(arrays, key => value)
      else
         push!(info, key => value)
      end
   end

   # move Atoms fields into `arrays` sub-dict
   # first X and Z, which should always be initialised
   arrays["pos"] = _write_convert(pop!(dict, "X"))
   arrays["species"] = _write_convert(string.(chemical_symbol.(atoms.Z)))
   arrays["Z"] = _write_convert(pop!(dict, "Z"))

   # now M and P, which might not be set
   for (source, dest) in zip(["M", "P"], ["masses", "momenta"])
      if source ∈ keys(dict) && dict[source] !== nothing && length(dict[source]) == natoms
         arrays[dest] = _write_convert(pop!(dict, source))
      end
   end

   # tidy up top-level dict entries
   dict["info"] = info
   dict["arrays"] = arrays
   delete!(dict, "data") # remove `nothing` entry
   delete!(dict, "calc") # remove `nothing` entry
   delete!(dict, "__id__")

   return dict
end

# define conversions to be applied when reading from ExtXYZ files to JuLIP Atoms
_read_convert(value, ::Int) = value
_read_convert(value::AbstractMatrix, natoms::Int) = size(value, 2) == natoms ? vecs(value) : value

function _extxyz_dict_to_atoms(dict)
   dict["__id__"] = "JuLIP_Atoms"

   natoms = Int(pop!(dict, "N_atoms"))
   info = pop!(dict, "info")
   arrays = pop!(dict, "arrays")

   "pos" in keys(arrays) || error("arrays dictionary missing 'pos' entry containing positions")
   dict["X"] = _read_convert(pop!(arrays, "pos"), natoms)
   @assert length(dict["X"]) == natoms

   # atomic numbers and symbols
   dict["Z"] = "Z" in keys(arrays) ? pop!(arrays, "Z") : nothing
   if "species" in keys(arrays)
      species = _read_convert(pop!(arrays, "species"), natoms)
      Zsp = atomic_number.([Symbol(sp) for sp in species])
      if dict["Z"] !== nothing
         all(dict["Z"] .== Zsp) || error("inconsistent 'Z' and 'species' properties")
      else
         dict["Z"] = Zsp
      end
   end
   dict["Z"] === nothing && error("atomic numbers not defined - either 'Z' or 'species' must be present")

   # mass - lookup from atomic number if not present
   if "masses" in keys(arrays)
      dict["M"] = _read_convert(pop!(arrays, "masses"), natoms)
   else
      dict["M"] = [atomic_mass(z) for z in AtomicNumber.(dict["Z"])]
   end

   # momenta / velocities
   if "momenta" in keys(arrays)
      dict["P"] = _read_convert(pop!(arrays, "momenta"), natoms)
   else
      dict["P"] = _read_convert(zeros((3, natoms)), natoms)
   end

   # add default for periodic boundary conditions (issue #151)
   "pbc" ∉ keys(dict) && (dict["pbc"] = [true, true, true])
   atoms = read_dict(dict)

   # everything else goes in data
   for (label, D) in zip((:info, :arrays), (info, arrays))
      for (key, value) in D
            value = _read_convert(value, natoms)
            set_data!(atoms, key, value)
      end
   end
   return atoms
end


"""
   read_extxyz(file, atoms[, range])

Read from an extended XYZ file using ExtXYZ.jl. 

`file` can be a string filename, open IO stream or `FILE*` pointer.
`range` can be absent, an integer frame number, integer range, or array of integers. 
Default behaviour is to read all frames in file.

Returns a vector of `Atoms` structs
"""
function read_extxyz(file, args...; kwargs...)
   results = Atoms[]
   for dict in ExtXYZ.read_frames(file, args...; kwargs...)
      push!(results, _extxyz_dict_to_atoms(dict))
   end
   return results
end

"""
   write_extxyz(file, atoms)

Write atoms to file in extended XYZ format using `ExtXYZ.jl`. 

`file` can be a string filename, open IO stream or `FILE*` pointer.
`atoms` can be either a single `JuLIP.Atoms` or a vector of `JuLIP.Atoms`.
"""
write_extxyz(file, atoms::Atoms{T}) where T = ExtXYZ.write_frame(file, _atoms_to_extxyz_dict(atoms))

write_extxyz(file, atoms::Vector{Atoms{T}}) where T = ExtXYZ.write_frames(file, _atoms_to_extxyz_dict.(atoms))

export read_extxyz, write_extxyz