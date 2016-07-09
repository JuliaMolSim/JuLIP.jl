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
