
"""
## module MatSciPy

### Summary

Julia wrappers for the [matscipy](https://github.com/libAtoms/matscipy) library.
At the moment, only the neighbourlist is wrapped, which is important for
fast neighbour calculations. `matscipy` depends on `ASE`, hence `MatSciPy.jl`
is a sub-module of `ASE.jl`.
For the most part it remains shielded from the user: if `ASE` detects that
`matscipy` is installed it will use `MatSciPy.NeighbourList`, otherwise it
will use `ASE.NeighbourList`. (TODO)

* `JuLIP.ASE.MatSciPy.neighbour_list` : the raw neighbour_list wrapper
* `JuLIP.ASE.MatSciPy.NeighbourList` : boiler plate type, including iterators
"""
module MatSciPy

using PyCall
@pyimport matscipy.neighbours as matscipy_neighbours

using JuLIP:  AbstractNeighbourList, cutoff, JVecs, vecs, pyarrayref
using JuLIP.ASE: ASEAtoms, pyobject

import JuLIP: sites, bonds
# to implement the iterators
import Base: start, done, next


"""
`neighbour_list(atoms::ASEAtoms, cutoff::Float64, quantities::AbstractString)`
`neighbour_list(atoms::ASEAtoms, cutoff::Float64)`

For the second version, `quantities = "ijdDS"`;
The elements of the tuple depend on the content of `quantities`. E.g.,
```
    i, j, D, d = neighbours(at, 5.0, "ijDd")
```
will return a vector `i` of atom indices, a vector of neighbour indices `j`,
the distance vectors in `D` and the scalar distances in `d`.

By convention, the string *must* start with `"ij"`.

**Warning:** to minimise overhead, this does *not* automatically convert the relative
distance vectors `D` from the ASE N x 3 to the Atoms.jl 3 x N convention!
Use the kwarg `convertarrays=true` to do so.
"""
function neighbour_list(atoms::ASEAtoms,
                        cutoff::Float64,
                        quantities="ijdDS";
                        convertarrays=true)
   results = matscipy_neighbours.neighbour_list(quantities,
                                                pyobject(atoms),
                                                cutoff)
   results2 = pycall(matscipy_neighbours.neighbour_list,
                     NTuple{length(quantities), PyArray}, quantities,
                     pyobject(atoms), cutoff)
   if length(quantities) == 1; results = (results,); end
   results = collect(Any, results)
   results2 = collect(Any, results2)

   for ii = 1:3
      @assert results[ii] == pyarrayref(results2[ii], own=true)
   end

   R = results[4]
   R2 = results2[4]
   @show typeof(R), size(R)
   @show typeof(R2), size(R2)
   @assert R' == pyarrayref(R2, own=true)
   @assert results[5]' == pyarrayref(results2[5], own=true)
   println("Assertions passed")
   quit()

   for (idx, quantity) in enumerate(quantities)
      # convert the PyArray to a Julia Array
      # results[idx] = pyarrayref(results[idx], own=true)
      # convert indices to 1-based (Julia convention)
      if (quantity == 'i' || quantity == 'j')
         results[idx][:] += 1
         # @simd for n = 1:length(results[idx])
         #    @inbounds results[idx][n] += 1
         # end
      end
      # convert R ans S matrices to arrays of vectors
      if convertarrays
         if quantity == 'D'; results[idx] = vecs(results[idx]'); end
         if quantity == 'S'; results[idx] = vecs(results[idx]'); end
      end
   end
   return tuple(results...)
end



"""
A basic wrapper around the `neighbour_list` builder.

Initialise an empty neighbourlist using
```
nlist = NeighbourList(at, cutoff)
```
where `at::ASEAtoms`, `cutoff::Float64`.
"""
type NeighbourList <: AbstractNeighbourList
    cutoff::Float64
    i::Vector{Int32}
    j::Vector{Int32}
    r::Vector{Float64}
    R::JVecs{Float64}
    S::JVecs{Int32}
end

# default constructor from an ASEAtoms object
NeighbourList(at::ASEAtoms, cutoff::Float64) =
   NeighbourList(cutoff, neighbour_list(at, cutoff, convertarrays=true)... )

import Base.length
length(nlist::NeighbourList) = length(nlist.i)

######################################################################
#### implementation of some iterators

bonds(nlist::NeighbourList) = zip(nlist.i, nlist.j, nlist.r, nlist.R, nlist.S)

# iterator over sites
type Sites
   nlist::NeighbourList
end

sites(nlist::NeighbourList) = Sites(nlist)

type SiteItState
   s::Int   # site index
   b::Int   # bond index
end

# first index is the site index, second index is the index into the nlist
start(s::Sites) = SiteItState(0, 0)
done(s::Sites, state::SiteItState) = (state.b == length(s.nlist))

function next(s::Sites, state::SiteItState)
   state.s += 1
   m0 = state.b+1
   while state.b < length(s.nlist) && s.nlist.i[state.b+1] <= state.s
      state.b += 1
   end
   m1 = state.b
   return (state.s, view(s.nlist.j, m0:m1), view(s.nlist.r, m0:m1),
               view(s.nlist.R, m0:m1), view(s.nlist.S, m0:m1)), state
end


end
