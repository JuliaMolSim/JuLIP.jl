

"""
## module ASE

### Summary

Provides Julia wrappers for some of ASE's functionality. Currently:

* `ase.Atoms` becomes `type ASEAtoms`
* `ase.calculators.neighborlist.NeighborList`
  should become `ASENeighborList`, but it is ignored to use
  the `matscipy` neighbourlist instead.
  TODO: implement ASENeighbourList as backup in case `matscipy` is
  not available
* `Atoms.get_array` becomes `get_data`
* `Atoms.set_array` becomes `set_data!`
"""
module ASE

# the functions to be implemented
import JuLIP:
      positions, set_positions!,  unsafe_positions,  # ✓
      cell, set_cell!,             # ✓
      pbc, set_pbc!,               # ✓
      set_data!, get_data,         # ✓
      calculator, set_calculator!, # ✓
      constraint, set_constraint!, # ✓
      neighbourlist,                # ✓
      energy, forces, virial

import Base.length, Base.deleteat!         # ✓

# from arrayconversions:
using JuLIP: mat, vecs, JVecF, JVecs, JVecsF, JMatF, pyarrayref,
      AbstractAtoms, AbstractConstraint, NullConstraint,
      AbstractCalculator, NullCalculator, defm

# extra ASE functionality:
import Base.repeat         # ✓
export ASEAtoms,      # ✓
      repeat, rnn, chemical_symbols, ASECalculator, extend!
using PyCall

@pyimport ase.io as ase_io
@pyimport ase.atoms as ase_atoms
@pyimport ase.build as ase_build


#################################################################
###  Wrapper for ASE Atoms object and its basic functionality
################################################################


"""
`type ASEAtoms <: AbstractAtoms`

Julia wrapper for the ASE `Atoms` class.

## Constructors

* `ASEAtoms(s::AbstractString, X::JVecsF) -> at::ASEAtoms`

For internal usage there is also a constructor `ASEAtoms(po::PyObject)`
"""
type ASEAtoms <: AbstractAtoms
   po::PyObject       # ase.Atoms instance
   calc::AbstractCalculator
   cons::AbstractConstraint
end


ASEAtoms(po::PyObject) = ASEAtoms(po, NullCalculator(), NullConstraint())

function set_calculator!(at::ASEAtoms, calc::AbstractCalculator)
   at.calc = calc
   return at
end
calculator(at::ASEAtoms) = at.calc
function set_constraint!(at::ASEAtoms, cons::AbstractConstraint)
   at.cons = cons
   return at
end
constraint(at::ASEAtoms) = at.cons


ASEAtoms(s::AbstractString, X::JVecsF) = ASEAtoms(s, mat(X))
ASEAtoms(s::AbstractString, X::Matrix) = ASEAtoms(ase_atoms.Atoms(s, X'))
ASEAtoms(s::AbstractString) = ASEAtoms(ase_atoms.Atoms(s))

"Return the PyObject associated with `a`"
pyobject(a::ASEAtoms) = a.po

get_array(a::ASEAtoms, name) = a.po[:get_array(name)]

set_array!(a::ASEAtoms, name, value) = a.po[:set_array(name, value)]

#
# TODO: write an explanation about storage layout here
#
positions(at::ASEAtoms) = copy( pyarrayref(at.po["positions"]) ) |> vecs
unsafe_positions(at::ASEAtoms) = pyarrayref(at.po["positions"]) |> vecs

function set_positions!(a::ASEAtoms, p::JVecsF)
   p_py = PyReverseDims(mat(p))
   a.po[:set_positions](p_py)
   return a
end

length(at::ASEAtoms) = length( unsafe_positions(at::ASEAtoms) )

set_pbc!(at::ASEAtoms, val::Bool) = set_pbc!(at, (val,val,val))

function set_pbc!(a::ASEAtoms, val::NTuple{3,Bool})
   a.po[:pbc] = val
   return a
end

pbc(a::ASEAtoms) = a.po[:pbc]

cell(at::ASEAtoms) = at.po[:get_cell]()

set_cell!(a::ASEAtoms, p::Matrix) = (a.po[:set_cell](p); a)

function deleteat!(at::ASEAtoms, n::Integer)
   at.po[:__delitem__](n-1) # delete in the actual array
   return at
end




"""
`repeat(a::ASEAtoms, n::(Int64, Int64, Int64)) -> ASEAtoms`

Takes an `ASEAtoms` configuration / cell and repeats is n_j times
into the j-th dimension. For example,
```
    atm = repeat( bulk("C"), (3,3,3) )
```
creates 3 x 3 x 3 unit cells of carbon.
"""
repeat(a::ASEAtoms, n::NTuple{3, Int64}) = ASEAtoms(a.po[:repeat](n))

import Base.*
*(at::ASEAtoms, n::NTuple{3, Int64}) = repeat(at, n)
*(n::NTuple{3, Int64}, at::ASEAtoms) = repeat(at, n)
*(at::ASEAtoms, n::Integer) = repeat(at, (n,n,n))
*(n::Integer, at::ASEAtoms) = repeat(at, (n,n,n))


export bulk, graphene_nanoribbon, nanotube, molecule

@doc ase_build.bulk[:__doc__] ->
bulk(args...; kwargs...) = ASEAtoms(ase_build.bulk(args...; kwargs...))

@doc ase_build.graphene_nanoribbon[:__doc__] ->
graphene_nanoribbon(args...; kwargs...) =
   ASEAtoms(ase_build.graphene_nanoribbon(args...; kwargs...))

"nanotube(n, m, length=1, bond=1.42, symbol=\"C\", verbose=False)"
nanotube(args...; kwargs...) =
      ASEAtoms(ase_build.nanotube(args...; kwargs...))

@doc ase_build.molecule[:__doc__] ->
      molecule(args...; kwargs...) =
         ASEAtoms(ase_build.molecule(args...; kwargs...))


############################################################
# matscipy neighbourlist functionality
############################################################

include("MatSciPy.jl")

neighbourlist(at::ASEAtoms, cutoff::Float64) = MatSciPy.NeighbourList(at, cutoff)



######################################################
#    Attaching an ASE-style calculator
#    ASE-style aliases
######################################################

type ASECalculator <: AbstractCalculator
   po::PyObject
end

function set_calculator!(at::ASEAtoms, calc::ASECalculator)
   at.po[:set_calculator](calc.po)
   at.calc = calc
   return at
end

set_calculator!(at::ASEAtoms, po::PyObject) =
      set_calculator!(at, ASECalculator(po))

forces(calc::ASECalculator, at::ASEAtoms) = calc.po[:get_forces](at.po)' |> vecs
energy(calc::ASECalculator, at::ASEAtoms) = calc.po[:get_potential_energy](at.po)

function virial(calc::ASECalculator, at::ASEAtoms)
    s = calc.po[:get_stress](at.po)
    vol = det(defm(at))
    if size(s) == (6,)
      # unpack stress from compressed Voigt vector form
      s11, s22, s33, s23, s13, s12 = s
      return -JMatF([s11 s12 s13;
                     s12 s22 s23;
                     s13 s23 s33]) * vol
    elseif size(s) == (3,3)
      return -JMatF(s) * vol
    else
      error("got unxpected size(stress) $(size(stress)) from ASE")
    end
end

"""
Creates an `ASECalculator` that uses `ase.calculators.emt` to compute
energy and forces. This is very slow and is only included for
demonstration purposes.
"""
function EMTCalculator()
   @pyimport ase.calculators.emt as emt
   return ASECalculator(emt.EMT())
end


################### extra ASE functionality  ###################
# TODO: we probably want more of this
#       and a more structured way to translate
#

"""
`extend!(at::ASEAtoms, atadd::ASEAtoms)`

add `atadd` atoms to `at` and returns `at`; only `at` is modified

A short variant is
```julia
extend!(at, (s, x))
```
where `s` is a string, `x::JVecF` a position
"""
function extend!(at::ASEAtoms, atadd::ASEAtoms)
   at.po[:extend](atadd.po)
   return at
end

extend!{S <: AbstractString}(at::ASEAtoms, atnew::Tuple{S,JVecF}) =
   extend!(at, ASEAtoms(atnew[1], atnew[2]))



"return vector of chemical symbols as strings"
chemical_symbols(at::ASEAtoms) = pyobject(at)[:get_chemical_symbols]()

write(filename::AbstractString, at::ASEAtoms) = ase_io.write(filename, at.po)

# TODO: trajectory; see
#       https://wiki.fysik.dtu.dk/ase/ase/io/trajectory.html


# TODO: rnn should be generalised to compute a reasonable rnn estimate for
#       an arbitrary set of positions
"""
`rnn(species)` : returns the nearest-neighbour distance for a given species
"""
function rnn(species::AbstractString)
   X = unsafe_positions(bulk(species) * 2)
   return minimum( norm(X[n]-X[m]) for n = 1:length(X) for m = n+1:length(X) )
end


end # module
