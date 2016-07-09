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
* `ase.calculators.neighborlist.NeighborList` becomes `ASENeighborList`

"""
module ASE

# the functions to be implemented
import JuLIP: AbstractAtoms,
      positions, set_positions!,   # ✓
      cell, set_cell!,             # ✓
      pbc, set_pbc!,               # ✓
      set_data!, get_data,         # ✓
      deleteat!,                   # ✓
      neighbourlist

import Base.length      # ✓

# from arrayconversions:
import JuLIP: mat, pts, vecs

# extra ASE functionality:
import Base.repeat         # ✓
export bulk, ASEAtoms      # ✓


# ASEAtoms, pyobject
# export convert, get_array, set_array!
# export get_cell, set_cell!
# export set_calculator, get_forces, get_potential_energy, get_stress
# export repeat, bulk, length
# export ASENeighborList, get_neighbors, neighbors
# export get_cell, cell, set_pbc!, iscubic, assert_cubic, delete_atom!


using PyCall
@pyimport ase
@pyimport ase.lattice as lattice
@pyimport ase.calculators.neighborlist as ase_neiglist



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
    po::PyObject      # ase.Atoms instance
    X::JPts
end


function ASEAtoms(s::AbstractString; repeat=nothing, cubic=false,
                  pbc=(true,true,true))
   at = bulk(s, cubic=cubic)
   repeat != nothing && at = repeat(at, repeat)
   set_pbc!(at, pbc)
end

ASEAtoms(po::PyObject) = ASEAtoms(po, positions(po))

"Return the PyObject associated with `a`"
pyobject(a::ASEAtoms) = a.po

get_array(a::ASEAtoms, name) = a.po[:get_array(name)]

set_array!(a::ASEAtoms, name, value) = a.po[:set_array(name, value)]

positions(po::PyObject) = po[:get_positions]()'

positions(a::ASEAtoms) = pts(a.X)

function set_positions!(a::ASEAtoms, p::JPts)
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
bulk(name::AbstractString; kwargs...) = ASEAtoms(lattice.bulk(name; kwargs...))



############################################################
##### TODO
############################################################

# TODO: tie in properly with ASE calculators
set_calculator!(a::ASEAtoms, calculator::PyObject) = a.po[:set_calculator](calculator)
get_forces(a::ASEAtoms) = a.po[:get_forces]()
get_potential_energy(a::ASEAtoms) = a.po[:get_potential_energy]()
get_stress(a::ASEAtoms) = a.po[:get_stress]()





############################################################
###  ASE Neighborlist implementation
############################################################


"""### type ASENeighborList <: AbstractNeighborList

This makes available the functionality of the ASE neighborlist implementation.
The neighborlist itself is actually stored in `ASEAtoms.po`, but
attaching an `ASENeighborList` will indicate this.

#### Keyword arguments:

 * skin = 0.3
 * sorted
 * self_interaction
 * bothways
"""
type ASENeighborList #  <: AbstractNeighborList
    po::PyObject
    cutoffs::Vector{Float64}
end

# # default constructor from a list of cut-offs
# ASENeighborList(cutoffs::Vector{Float64}; kwargs...) =
#     ASENeighborList(ase_neiglist.NeighborList(cutoffs; kwargs...))

# constructor from an ASEAtoms object, with a single cut-off
# this generates a list of cut-offs, then the neighborulist, then
# builds the neighborlist from the ASEAtoms object
ASENeighborList(atm::ASEAtoms, cutoff::Float64; kwargs...) =
    ASENeighborList(atm, cutoff * ones(length(atm)); kwargs...)

# constructor from an ASEAtoms object, with multiple cutoffs
# this also builds the neighborlist
function ASENeighborList(atm::ASEAtoms, cutoffs::Vector{Float64};
                         bothways=true, self_interaction=false, kwargs...)
    po = ase_neiglist.NeighborList(cutoffs * 0.5;
                                   bothways=bothways,
                                   self_interaction=self_interaction,
                                   kwargs...)
    nlist = ASENeighborList(po, cutoffs)
    update!(nlist, atm)
    return nlist
end

# # regain the cutoffs vector
# _get_cutoffs_ref_(nlist::ASENeighborList) =
#     PyArray(nlist.po["cutoffs"])


"""`update!(nlist::ASENeighborList, atm::ASEAtoms)`

checks whether the atom positions have moved by more than the skin
and rebuilds the list if they have done so.
"""
update!(nlist::ASENeighborList, atm::ASEAtoms) =  nlist.po[:update](atm.po)


"""`build!(nlist::ASENeighborList, atm::ASEAtoms)`

force rebuild of the neighborlist
"""
build!(nlist::ASENeighborList, atm::ASEAtoms) = nlist.po[:build](atm.po)


"""`get_neighbors(n::Integer, neiglist::ASENeighborList) -> (indices, offsets)`

Return neighbors and offsets of atom number n. The positions of the neighbor
atoms can then calculated like this: (python code)

```
          indices, offsets = nl.get_neighbors(42)
          for i, offset in zip(indices, offsets):
              print(atoms.positions[i] + dot(offset, atoms.get_cell()))
```

If `get_neighbors(a)` gives atom b as a neighbor,
    then `get_neighbors(b)` will not return a as a neighbor, unless
    `bothways=True` was used.
"""
function get_neighbors(n::Integer, neiglist::ASENeighborList)
    indices, offset = neiglist.po[:get_neighbors](n-1)
    indices .+= 1
    return (indices, offset)
end
# "alias for `get_neighbors`"
# get_neighbors = get_neighbors
# "alias for `get_neighbors`"
# neighbors = get_neighbors
"alias for `get_neighbors`"
neighbors = get_neighbors


"""`get_neighbors(n::Integer, neiglist::ASENeighborList, atm::ASEAtoms)
              -> (indices::Vector{Int}, s::Vector{Float64}, r::Matrix{Float64})`

 * `indices`: indices of atom positions
 * `s`: scalar distances of neighbours
 * `r`: relative positions of neighbours (vectorial dist)

This is a convenience function that does some of the work of constructing the
neighborhood. This is probably fairly inefficient to use since it has
to construct a `PyArray` object for the positions every time it is called.
Instead, use the iterator. Problem is, this is not much faster, because
the PyCall conversion overhead is so horrendous.
"""
function get_neighbors(n::Integer, neiglist::ASENeighborList, atm::ASEAtoms;
                       rcut=Inf)
    (inds, offsets) = ASE.neighbors(n, neiglist)
    X = _get_positions_ref_(atm)
    cell = get_cell(atm)
    r = X[inds, :]' + cell' * offsets' .- slice(X, n, :)
    s = sqrt(sumabs2(r, 1))
    I = find(s .<= neiglist.cutoffs[n])
    return inds[I], s[I], r[:, I]
end



##################### NEIGHBOURHOOD ITERATOR ###################
# this is about twice as fast as `get_neighbours`
# which indicates that, either, the ASE neighbour list is very slow
#   or, the overhead from the python call is horrendous!
#

type ASEAtomIteratorState
    at::ASEAtoms
    neiglist::ASENeighborList
    n::Int         # iteration index
    X::Matrix{Float64}
    cell_t::Matrix{Float64}
end

ASEAtomIteratorState(at::ASEAtoms, neiglist::ASENeighborList) =
    ASEAtomIteratorState( at, neiglist, 0,
                          get_positions(at),
                          get_cell(at)' )

import Base.start
start(I::Tuple{ASEAtoms,ASENeighborList}) =
    ASEAtomIteratorState(I...)

import Base.done
done(I::Tuple{ASEAtoms,ASENeighborList}, state::ASEAtomIteratorState) =
    (state.n == length(state.at))

import Base.next
function next(I::Tuple{ASEAtoms,ASENeighborList}, state::ASEAtomIteratorState)
    state.n += 1
    inds, offsets = neighbors(state.n, state.neiglist)
    r = state.X[:,inds] + state.cell_t * offsets' .- state.X[:,state.n]
    s = sqrt(sumabs2(r, 1))
    I = find(s .<= I[2].cutoffs[state.n])
    return (state.n, inds, s, r), state
    # now find the indices that are actually within the cut-off
    #  TODO: resolve the nasty issue of 1/2 cut-off first?!?!?
    # I = find(s .< 3.0)   # state.cutoffs[state.n])  (hard-coded for debugging)
    # return (state.n, inds[I], s[I], r[:,I]), state
end



################### some useful hacks ###################
#
#

export rnn

"""
`rnn(species)` :
computes the nearest-neighbour distance for a given species
"""
function rnn(species)
    at = bulk(species)
    at = repeat(at, (2,2,2))
    X = positions(at)
    R = zeros(length(at), length(at))
    for n = 1:length(at)
        for m = n+1:length(at)
            R[n,m] = norm(X[:,n] - X[:,m])
        end
    end
    R[find(R .== 0)] = maximum(R[:])
    rnn = minimum(R[:])
end

end
