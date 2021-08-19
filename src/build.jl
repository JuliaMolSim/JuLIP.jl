
"""
`module Build` : this is a very poor man's version of `ase.build`. At the moment
only a subset of the `bulk` functionality is supported.
"""
module Build

import JuLIP
using ..Chemistry
using JuLIP: JVec, JMat, JVecF, JMatF, mat, vecs,
      Atoms, cell, cell_vecs, positions, momenta, masses, numbers, pbc,
      chemical_symbols, set_cell!, set_pbc!, update_data!,
      apply_defm!, calculator, set_calculator!

using LinearAlgebra: I, Diagonal, isdiag, norm, cross, det

import Base: union
import JuLIP: Atoms

export repeat, bulk, cluster, autocell!, append, rotation_matrix, rotate!




_auto_pbc(pbc::Tuple{Bool, Bool, Bool}) = JVec(pbc...)
_auto_pbc(pbc::Bool) = JVec(pbc, pbc, pbc)
_auto_pbc(pbc::AbstractVector) = JVec(pbc...)

_auto_cell(cell) = cell
_auto_cell(cell::AbstractMatrix) = JMat(cell)
_auto_cell(C::AbstractVector{T}) where {T <: Real} = (
   length(C) == 3 ? JMatF(C[1], 0.0, 0.0, 0.0, C[2], 0.0, 0.0, 0.0, C[3]) :
                    JMatF(C...) )  # (case 2 requires that length(C) == 0
_auto_cell(C::AbstractVector{T}) where {T <: AbstractVector} = JMatF([ C[1] C[2] C[3] ])
_auto_cell(C::AbstractVector) = _auto_cell([ c for c in C ])

"""
`autocell!(at::Atoms) -> Atoms`

generates cell vectors that contain all atom positions + a small buffer,
sets the cell of `at` accordingly (in-place) and returns the same atoms objet
with the new cell.
"""
autocell!(at::Atoms) = set_cell!(at, _autocell(positions(at)))

function _autocell(X::AbstractVector{JVec{T}}) where T
   ext = extrema(mat(X), dims = (2,))
   C = Diagonal([ e[2] - e[1] + 1.0  for e in ext ][:])
   return JMat{T}(C)
end

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

_auto_M(M::AbstractVector{T}) where {T <: AbstractFloat} = Vector{T}(M)
_auto_M(M::AbstractVector) = Vector{Float64}(M)

_auto_Z(Z::AbstractVector) = Vector{AtomicNumber}(Z)

const PositionArray{T} = Union{AbstractMatrix{T}, AbstractVector{JVec{T}}}

Atoms(; kwargs...) = Atoms{Float64}(; kwargs...)

Atoms{T}(; X = JVec{T}[],
           P = JVec{T}[],
           M = T[],
           Z = AtomicNumber[],
           cell = zero(JMat{T}),
           pbc = JVec(false, false, false),
           calc = nothing ) where {T} =
      Atoms(X, P, M, Z, cell, pbc, calc)

Atoms(X, P, M, Z, cell, pbc, calc=nothing) =
      Atoms(_auto_X(X), _auto_X(P), _auto_M(M), _auto_Z(Z),
            _auto_cell(cell), _auto_pbc(pbc), calc)

Atoms(Z::Vector{AtomicNumber}, X::Vector{JVec{T}}; kwargs...) where {T} =
  Atoms{T}(; Z=Z, X=X, kwargs...)


Atoms( Z::Vector{<: Integer}, X::PositionArray{T}; kwargs...
       ) where {T <: AbstractFloat} =
   Atoms{T}(; Z=Z, X=X, kwargs...)

"""
simple way to construct an atoms object from just positions
"""
Atoms(s::Symbol, X::PositionArray{T}; kwargs...) where {T <: AbstractFloat} =
      Atoms{T}(; X = X,
                 M = fill(one(T), length(X)),
                 P = zeros(JVec{T}, length(X)),
                 Z = fill(atomic_number(s), length(X)),
                 cell = _autocell(X),
                 pbc = (false, false, false), kwargs... )

Atoms(s::Symbol, X::Matrix{Float64}) = Atoms(s, vecs(X))

# shallow copy constructor
Atoms(at::Atoms) = Atoms(at.X, at.P, at.M, at.Z, at.cell, at.pbc, at.calc)

import Base.copy

"""
   at2 = copy(at)

Return a copy of Atoms, referring to same arrays for positions, momenta, etc.
Use `deepcopy()` for a deep copy of the arrays as well.
"""
copy(at::Atoms) = Atoms(at)

"""
   rotation_matrix([x=X, y=Y, z=Z])

Construct a rotation matrix from two or more Miller indicies. 

Example usage:

```
A = rotation_matrix(x=[1,1,1], y=[1,-1,0]) # fills in z from cross(x, y)
```
"""
function rotation_matrix(; x=nothing, y=nothing, z=nothing)
   @assert sum([x, y, z] .!== nothing) >= 2

   x !== nothing && (x /= norm(x))
   y !== nothing && (y /= norm(y))
   z !== nothing && (z /= norm(z))

   if x === nothing
      @assert isapprox(y' * z, 0, atol=1e-6)
      x = cross(y, z)
      x /= norm(x)
   end

   if y === nothing
      @assert isapprox(z' * x, 0, atol=1e-6)
      y = cross(z, x)
      y /= norm(y)
   end

   if z === nothing
      @assert isapprox.(x' * y, 0, atol=1e-6)
      z = cross(x, y)
      z /= norm(z)
   end

   A = hcat(x, y, z)
   (det(A) < 0.0) && (A = hcat(-x, y, z))
   return A
end

"""
Rotate atoms in place to align axes with Miller indices x, y, z, at least two of which must be given.
"""
rotate!(at::Atoms; x=nothing, y=nothing, z=nothing) = apply_defm!(at, rotation_matrix(x=x, y=y, z=z))


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



function _simple_bulk(sym::Symbol, cubic::Bool; a=nothing)
   if cubic
      X, scal = _cubic_cells[symmetry(sym)]
      C = Matrix(1.0I, 3, 3) / scal
   else
      X, C, scal = _unit_cells[symmetry(sym)]
   end
   a === nothing && (a = lattice_constant(sym))
   return [ JVecF(x) * a * scal  for x in X ], JMatF(C * a * scal)
end


function _bulk_hcp(sym::Symbol; a=nothing, c=nothing)
   D = Chemistry.refstate(sym)
   a === nothing && (a = D["a"])
   c === nothing && (c = a * D["c/a"])
   return [JVecF(0.0, 0.0, 0.0), JVecF(0.0, a / sqrt(3), c / 2)],
          JMatF( [a, -a / 2,  0.0, 0.0, a * sqrt(3)/2, 0.0, 0.0, 0.0, c] )
end


_convert_pbc(pbc::NTuple{3, Bool}) = pbc
_convert_pbc(pbc::Bool) = (pbc, pbc, pbc)

function bulk(sym::Symbol; T=Float64, cubic = false, pbc = (true,true,true), a=nothing, c=nothing, x=nothing, y=nothing, z=nothing)
   symm = symmetry(sym)
   if symm in _simple_structures
      X, C = _simple_bulk(sym, cubic; a=a)
   elseif symm == :hcp
      X, C = _bulk_hcp(sym; a=a, c=c)  # cubic parameter is irrelevant for hcp
   end
   m = atomic_mass(sym)
   Z = atomic_number(sym)
   nat = length(X)
   at = Atoms( convert(Vector{JVec{T}}, X),
                 fill(zero(JVec{T}), nat),
                 fill(T(m), nat),
                 fill(AtomicNumber(Z), nat),
                 convert(JMat{T}, C),
                 _convert_pbc(pbc)  )
   (x !== nothing || y !== nothing || z !== nothing) && rotate!(at, x=x, y=y, z=z)
   return at
end


"""
auxiliary function to convert a general cell to a cubic one; this is a bit of
a hack, so to not make a mess of things, this will only work in very restrictive
circumstances and otherwise throw an error. But it could be revisited.
"""
function _cubic_cell(atu::Atoms{T}) where {T}
   @assert length(atu) == 1
   ru = JuLIP.rmin(atu)
   at = bulk(chemical_symbol(atu.Z[1]), cubic=true)
   r = JuLIP.rmin(at)
   return apply_defm!(at, (ru/r) * one(JMat{T}))
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
   `[1,2,3]` for 3D point defects and `[1,2]` for 2D dislocations, in the
   remaining dimension(s) the b.c. will be periodic.
* `shape` : shape of the cluster, at the moment only `:ball` is allowed
* `parity`: enforce an odd (where `parity=1`) or even (where `parity=0`)
   number of cells in each lattice direction. Use a `nothing` element to
   leave the number of cells unchanged (e.g. in periodic directions).
## todo
 * lift the restriction of single species
 * allow other shapes
"""
function cluster(atu::Atoms{T}, R::Real;
                 dims = findall(pbc(atu).==true),
                 shape = :ball, x0=nothing,
                 parity=nothing) where {T}
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
   if parity !== nothing
      L[((L .% 2) .!= parity) .& (parity .!= nothing)] .+= 1
   end
   # multiply
   at = atu * L
   # find point closest to centre
   x̄ = sum( x[dims] for x in at.X ) / length(at.X)
   i0 = findmin( [norm(x[dims] - x̄) for x in at.X] )[2]
   if x0 === nothing
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
   at1 = Atoms(X, P, M, Z, Diagonal(collect(n)) * cell(at), pbc(at))
   set_calculator!(at1, calculator(at))
   return at1
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
   JuLIP.reset_clamp!(at)
   return at
end



union(at1::Atoms, at2::Atoms) =
   Atoms( X = union(at1.X, at2.X),
          P = union(at1.P, at2.P),
          M = union(at1.M, at2.M),
          Z = union(at1.Z, at2.Z),
          cell = cell(at1),
          pbc = pbc(at1) )

append(at::Atoms, X::AbstractVector{<:JVec}) =
   Atoms( X = union(at.X, X),
          P = union(at.P, zeros(JVecF, length(X))),
          M = union(at.M, zeros(length(X))),
          Z = union(at.Z, zeros(Int, length(X))),
          cell = cell(at),
          pbc = pbc(at) )




end
