###
#
# TODO
#   * add links to key arrays so they can be modified in-place
#     or accessed cheaply
#
#   * in all my julia codes I use self_interaction=false and bothways=true
#      - is it a bad idea if I make these default?
#
#   * rcut in ASE denotes spheres of overlap?!?!? i.e. it is in effect
#     half of the cut-off of the potential?


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
import JuLIP: AbstractAtoms,
      positions, set_positions!,   # ✓
      cell, set_cell!,             # ✓
      pbc, set_pbc!,               # ✓
      set_data!, get_data,         # ✓
      deleteat!,                   # ✓
      neighbourlist                # ✓

import Base.length      # ✓

# from arrayconversions:
import JuLIP: mat, pts, vecs, JPts, JVecs

# extra ASE functionality:
import Base.repeat         # ✓
export bulk, ASEAtoms      # ✓

# this one is a little hack based on the ASE functionality, hence it is not
# in JuLIP proper.
export rnn                 # TODO: double-check and test


using PyCall
@pyimport ase
@pyimport ase.lattice as lattice
# @pyimport ase.calculators.neighborlist as ase_neiglist



#################################################################
###  Wrapper for ASE Atoms object and its basic functionality
################################################################


"""
`type ASEAtoms <: AbstractAtoms`

Julia wrapper for the ASE `Atoms` class.

## Constructors

The default constructor is
```
at = ASEAtoms("Al"; repeat=(2,3,4), cubic=true, pbc = (true, false, true))
```

For internal usage there is also a constructor `ASEAtoms(po::PyObject)`
"""
type ASEAtoms <: AbstractAtoms
    po::PyObject       # ase.Atoms instance
    X::JPts{Float64}   # an alias for positions, for faster access
end


ASEAtoms(po::PyObject) = ASEAtoms(po, positions(po))


function ASEAtoms( s::AbstractString;
                   repeatcell=nothing, cubic=false, pbc=(true,true,true) )
   at = bulk(s, cubic=cubic)
   if repeatcell != nothing; at = repeat(at, repeatcell); end
   set_pbc!(at, pbc)
   return at
end


"Return the PyObject associated with `a`"
pyobject(a::ASEAtoms) = a.po

get_array(a::ASEAtoms, name) = a.po[:get_array(name)]

set_array!(a::ASEAtoms, name, value) = a.po[:set_array(name, value)]

positions(po::PyObject) = pts(po[:get_positions]()')

positions(a::ASEAtoms) = a.X

function set_positions!(a::ASEAtoms, p::JPts{Float64})
   a.X = p
   a.po[:set_positions](mat(p)')
   return nothing
end

length(a::ASEAtoms) = length(a.X)

set_pbc!(a::ASEAtoms, val) = (a.po[:pbc] = val)

pbc(a::ASEAtoms) = a.po[:pbc]

cell(at::ASEAtoms) = at.po[:get_cell]()

set_cell!(a::ASEAtoms, p::Matrix) = a.po[:set_cell](p)

function deleteat!(a::ASEAtoms, n::Integer)
   at.po[:__delitem__](n-1) # delete in the actual array
   deleteat!(a.X, n)        # delete in alias a.X
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

"""
`bulk(name::AbstractString; kwargs...) -> ASEAtoms`

Generates a unit cell of the element described by `name`
"""
bulk(name::AbstractString; kwargs...) =
            ASEAtoms(lattice.bulk(name; kwargs...))


############################################################
# matscipy neighbourlist functionality
############################################################

include("matscipy.jl")

neighbourlist(at::ASEAtoms, cutoff::Float64) = MatSciPy.NeighbourList(at, cutoff)


######################################################
##### TODO
############################################################

# TODO: tie in properly with ASE calculators
set_calculator!(a::ASEAtoms, calculator::PyObject) = a.po[:set_calculator](calculator)
get_forces(a::ASEAtoms) = a.po[:get_forces]()
get_potential_energy(a::ASEAtoms) = a.po[:get_potential_energy]()
get_stress(a::ASEAtoms) = a.po[:get_stress]()








################### some additional hacks ###################


"""
`rnn(species)` : returns the nearest-neighbour distance for a given species
"""
function rnn(species::AbstractString)
   at = ASEAtoms(species, repeat=(2,2,2))
   X = positions(at)
   r = Float64[]
   for n = 1:length(at), m = n+1:length(at)
      push!(r, norm(X[m]-X[n]))
   end
   return minimum(r)
end



end
