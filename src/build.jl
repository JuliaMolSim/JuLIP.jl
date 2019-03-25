
"""
`module Build` : this is a very poor man's version of `ase.build`. At the moment
only a subset of the `bulk` functionality is supported.
"""
module Build

import JuLIP
using ..Chemistry
using JuLIP: JVec, JMat, JVecF, JMatF, JVecsF, mat,
      Atoms, cell, cell_vecs, positions, momenta, masses, numbers, pbc,
      chemical_symbols, set_cell!, set_pbc!, update_data!,
      set_defm!, defm

using LinearAlgebra: I, Diagonal, isdiag, norm

import Base: union

export repeat, bulk, cluster, autocell!, append





const _unit_cells = Dict(     # (positions, cell matrix, factor of a)
   :fcc => ( [ [0 0 0], ],
             [0 1 1; 1 0 1; 1 1 0],  0.5),
   :bcc => ( [ [0.0,0.0,0.0], ],
             [-1 1 1; 1 -1 1; 1 1 -1], 0.5),
   :diamond => ( [ [0.0, 0.0, 0.0], [0.5, 0.5, 0.5] ],
                 [0 1 1; 1 0 1; 1 1 0], 0.5)
)

const _cubic_cells = Dict(   # (positions, factor of a)
   :fcc => ( [ [0 0 0], [0 1 1], [1 0 1], [1 1 0] ], 0.5 ),
   :bcc => ( [ [0 0 0], [1 1 1] ], 0.5 ),
   :diamond => ( [ [0 0 0], [1 1 1], [0 2 2], [1 3 3], [2 0 2],
                   [3 1 3], [2 2 0], [3 3 1] ], 0.25 )
)

_simple_structures = [:fcc, :bcc, :diamond]



function _simple_bulk(sym::Symbol, cubic::Bool)
   if cubic
      X, scal = _cubic_cells[symmetry(sym)]
      C = Matrix(1.0I, 3, 3) / scal
   else
      X, C, scal = _unit_cells[symmetry(sym)]
   end
   a = lattice_constant(sym)
   return [ JVecF(x) * a * scal  for x in X ], JMatF(C * a * scal)
end


function _bulk_hcp(sym::Symbol)
   D = Chemistry.refstate(sym)
   a = D["a"]
   c = a * D["c/a"]
   return [JVecF(0.0, 0.0, 0.0), JVecF(0.0, a / sqrt(3), c / 2)],
          JMatF( [a, -a / 2,  0.0, 0.0, a * sqrt(3)/2, 0.0, 0.0, 0.0, c] )
end


_convert_pbc(pbc::NTuple{3, Bool}) = pbc
_convert_pbc(pbc::Bool) = (pbc, pbc, pbc)

function bulk(sym::Symbol; cubic = false, pbc = (true,true,true))
   symm = symmetry(sym)
   if symm in _simple_structures
      X, C = _simple_bulk(sym, cubic)
   elseif symm == :hcp
      X, C = _bulk_hcp(sym)  # cubic parameter is irrelevant for hcp
   end
   m = atomic_mass(sym)
   z = atomic_number(sym)
   nat = length(X)
   return Atoms( X, fill(zero(JVecF), nat), fill(m, nat), fill(z, nat), C,
                 _convert_pbc(pbc) )
end


"""
auxiliary function to convert a general cell to a cubic one; this is a bit of
a hack, so to not make a mess of things, this will only work in very restrictive
circumstances and otherwise throw an error. But it could be revisited.
"""
function _cubic_cell(atu::Atoms)
   @assert length(atu) == 1
   ru = JuLIP.rmin(atu)
   at = bulk(chemical_symbol(atu.Z[1]), cubic=true)
   r = JuLIP.rmin(at)
   return set_defm!(at, (ru/r) * defm(at); updatepositions=true)
end


"""
`cluster(args...; kwargs...) -> at::AbstractAtoms`

Produce a cluster of approximately radius R. The center
atom is always at index 1.

## Methods
```
cluster(atu::AbstractAtoms, R::Real)  # specify unit cell
cluster(sym::Symbol, R::Real)         # atu = bulk(sym; cubic=true)
```
Both methods assume that there is only a single species, and both require
the use of an orthorhombic unit cell (for now).

* `sym`: chemical symbol of the atomic species
* `R` : length-scale, meaning depends on shape of the cluster

## Keyword Arguments:
* `dims` : dimension into which the cluster is extended, typically
   `(1,2,3)` for 3D point defects and `(1,2)` for 2D dislocations, in the
   remaining dimension(s) the b.c. will be periodic.
* `shape` : shape of the cluster, at the moment only `:ball` is allowed

## TODO
 * lift the restriction of single species
 * allow other shapes
"""
function cluster(atu::Atoms{T}, R::Real; dims = [1,2,3], shape = :ball, x0=nothing) where T
   sym = chemical_symbols(atu)[1]
   # check that the cell is orthorombic
   if !isdiag(cell(atu))
      if length(atu) > 1
         error("""`JuLIP.cluster` requires as argument either a cubic cell or a
                  one-atom cell.""")
      end
      atu = _cubic_cell(atu)
   end
   # # check that the first index is the centre
   # @assert norm(atu[1]) == 0.0
   # determine by how much to multiply in each direction
   Fu = cell(atu)'
   L = [ j ∈ dims ? 2 * (ceil(Int, R/Fu[j,j])+3) : 1    for j = 1:3]
   # multiply
   at = Atoms(atu) * L
   # find point closest to centre
   x̄ = sum( x[dims] for x in at.X ) / length(at.X)
   i0 = findmin( [norm(x[dims] - x̄) for x in at.X] )[2]
   if x0 == nothing
      x0 = at[i0]
   else
      x0 += at[i0]
   end
   # swap positions
   X = positions(at)
   X[1], X[i0] = X[i0], X[1]
   F = Diagonal([Fu[j,j]*L[j] for j = 1:3])
   # carve out a cluster with mini-buffer to account for round-off
   r = [ norm(x[dims] - x0[dims]) for x in X ]
   IR = findall( r .<= R+sqrt(eps(T)) )
   # generate new positions
   Xcluster = X[IR]
   # generate the cluster
   at_cluster = Atoms(sym, Xcluster)
   set_cell!(at_cluster, F')
   # take open boundary in all directions specified in dims, periodic in the rest
   set_pbc!(at_cluster, [ !(j ∈ dims) for j = 1:3 ])
   # return the cluster and the index of the center atom
   return at_cluster
end


cluster(sym::Symbol, R::Real; kwargs...) =
      cluster( bulk(sym, cubic=true), R; kwargs... )



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
function Base.repeat(at::Atoms, n::NTuple{3})
   C0 = cell(at)
   c1, c2, c3 = cell_vecs(at)
   X0, P0, M0, Z0 = positions(at), momenta(at), masses(at), numbers(at)
   nrep = n[1] * n[2] * n[3]
   nat0 = length(at)
   X = Base.repeat(X0, outer=nrep)
   P = Base.repeat(P0, outer=nrep)
   M = Base.repeat(M0, outer=nrep)
   Z = Base.repeat(Z0, outer=nrep)
   i = 0
   for a in CartesianIndices( (1:n[1], 1:n[2], 1:n[3]) )
      b = c1 * (a[1]-1) + c2 * (a[2]-1) + c3 * (a[3]-1)
      X[i+1:i+nat0] = [b+x for x in X0]
      i += nat0
   end
   return Atoms(X, P, M, Z, Diagonal(collect(n)) * cell(at), pbc(at))
end

Base.repeat(at::Atoms, n::Integer) = repeat(at, (n,n,n))
Base.repeat(at::Atoms, n::AbstractArray) = repeat(at, (n...,))

import Base.*
*(at::Atoms, n) = repeat(at, n)
*(n, at::Atoms) = repeat(at, n)


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
simple way to construct an atoms object from just positions
"""
Atoms(s::Symbol, X::Vector{JVecF}) =
      Atoms( X, fill(zero(JVecF), length(X)), fill(atomic_mass(s), length(X)),
             fill(atomic_number(s), length(X)), _autocell(X),
             (false, false, false) )

Atoms(s::Symbol, X::Matrix{Float64}) = Atoms(s, vecs(X))


function _autocell(X::Vector{JVec{T}}) where T
   ext = extrema(mat(X), dims = (2,))
   C = Diagonal([ e[2] - e[1] + 1.0  for e in ext ][:])
   return JMat{T}(C)
end

"""
`autocell!(at::Atoms) -> Atoms`

generates cell vectors that contain all atom positions + a small buffer,
sets the cell of `at` accordingly (in-place) and returns the same atoms objet
with the new cell.
"""
autocell!(at::Atoms) = set_cell!(at, _autocell(positions(at)))


union(at1::Atoms, at2::Atoms) =
   Atoms( X = union(at1.X, at2.X),
          P = union(at1.P, at2.P),
          M = union(at1.M, at2.M),
          Z = union(at1.Z, at2.Z),
          cell = cell(at1),
          pbc = pbc(at1) )

append(at::Atoms, X::JVecsF) =
   Atoms( X = union(at.X, X),
          P = union(at.P, zeros(JVecF, length(X))),
          M = union(at.M, zeros(length(X))),
          Z = union(at.Z, zeros(Int, length(X))),
          cell = cell(at),
          pbc = pbc(at) )




end
