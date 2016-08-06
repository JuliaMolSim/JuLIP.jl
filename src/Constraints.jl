
"""
`module Constraints`

TODO: write documentation
"""
module Constraints

import JuLIP: Dofs, AbstractConstraint,
         dofs, project!, set_positions!, positions,
         mat, vecs, pts, JVecs, JPts, JVecsPts,
         AbstractAtoms

export FixedCell


function zeros_free{T}(n::Integer, x::Vector{T}, free::Vector{Int})
   z = zeros(T, n)
   z[free] = x
   return z
end

function insert_free!{T}(p::Array{T}, x::Vector{T}, free::Vector{Int})
   p[free] = x
   return p
end


"""
`FixedCell`: no constraints are placed on the motion of atoms, but the
cell shape is fixed

Constructor:
```julia
FixedCell(at::AbstractAtoms; free=..., clamp=..., mask=...)
```
Set at most one of the kwargs:
* no kwarg: all atoms are free
* `free` : list of free atom indices (not dof indices)
* `clamp` : list of clamped atom indices (not dof indices)
* `mask` : TODO
"""
type FixedCell <: AbstractConstraint
   ifree::Vector{Int}
end

function FixedCell(at::AbstractAtoms; free=false, clamp=false, mask=false)
   if any( (free, clamp, mask) )
      error("FixedCell: only one of `free`, `clamp`, `mask` may be provided")
   elseif all( (!free, !clamp, !mask) )
      # in this case (default) all atoms are free
      return FixedCell(collect(1:3*length(at)))
   end

   # determine free dof indices
   Nat = length(at)
   if clamp != false
      # reverse to setting free
      free = setdiff(1:length(at), clamp)
   end
   if free != false
      # convert free-at indices to free dof indices
      return FixedCell([free; Nat+free; 2*Nat+free][:])
   end
   @assert mask != false
   error("FixedCell: `mask` is not yet implemented")
end

dofs{T}( at::AbstractAtoms, cons::FixedCell, v::JVecs{T}) = mat(v)[cons.ifree]
dofs{T}( at::AbstractAtoms, cons::FixedCell, p::JPts{T}) = mat(p)[cons.ifree]

vecs(cons::FixedCell, at::AbstractAtoms, dofs::Dofs) =
      zeros_free(length(at), dofs, cons.ifree) |> vecs

positions(cons::FixedCell, at::AbstractAtoms, dofs::Dofs) =
      insert_free!(positions(at) |> mat, dofs, cons.ifree) |> pts

project!(cons::FixedCell, at::AbstractAtoms) = at

# TODO: this is a temporaruy hack, and I think we need to
#       figure out how to do this for more general constraints
project!(cons::FixedCell, A::SparseMatrixCSC) = A[cons.ifree, cons.ifree]


end
