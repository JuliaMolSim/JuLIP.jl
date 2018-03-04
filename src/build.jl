
"""
`module Build` : this is a very poor man's version of `ase.build`. At the moment
only a subset of the `bulk` functionality is supported.
"""
module Build

using ..Chemistry
using JuLIP: JVec, JMat, JVecF, JMatF, JVecsF,
      Atoms, cell, cell_vecs, positions, momenta, masses, numbers, pbc


export repeat, bulk




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
      C = eye(3) / scal
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
   for a in CartesianRange( CartesianIndex(1,1,1), CartesianIndex(n...) )
      b = c1 * (a[1]-1) + c2 * (a[2]-1) + c3 * (a[3]-1)
      X[i+1:i+nat0] = [b+x for x in X0]
      i += nat0
   end
   return Atoms(X, P, M, Z, JMat(diagm([n...]) * cell(at)), pbc(at))
end

Base.repeat(at::Atoms, n::Integer) = repeat(at, (n,n,n))

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
      Atoms( X, fill(JVecF, length(X)), fill(atomic_mass(s), length(X)),
             fill(atomic_number(s), length(X)), _autocell(X),
             (false, false, false) )
Atoms(s::Symbol, X::Matrix{Float64}) = Atoms(s, vecs(X))

function _autocell(X::JVecsF)
   ext = extrema(mat(JVecsF), 2)
   C = diagm([ e[2] - e[1] + 1.0  for e in ext ])
   return JMatF(C)
end


end
