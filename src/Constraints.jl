
"""
`module Constraints`

TODO: write documentation
"""
module Constraints

import JuLIP: Dofs, AbstractConstraint, dofs, project!, set_positions!,
         mat, vecs, JVecs, JPts,
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
      error("FixedCell: only one of free, clamp, mask may be provided")
   elseif all( (!free, !clamp, !mask) )
      free = collect(1:3*length(at))
   else
      error("TODO: fix the `FixedCell` constructor")
   end

   if free != nothing
      # TODO: convert free-at >>> free-dof
      return FixedCell(free)   # this is wrong!!!!
   elseif clamp != nothing
      free = setdiff(1:length(at), clamp)
      return FixedCell(setdiff(1:length(at), clamp))
   end
   # mask case
   error("FixedCell: `mask` is not yet implemented")
end

dofs(at::AbstractAtoms, cons::FixedCell, vorp::Union{JVecs,JPts}) = mat(vorp)[cons.ifree]

vecs(at::AbstractAtoms, cons::FixedCell, dofs::Dofs) =
      zeros_free(length(at), dofs, cons.ifree) |> vecs

positions(at::AbstractAtoms, cons::FixedCell, dofs::Dofs) =
      insert_free!(positions(at) |> mat, dofs, cons.ifree) |> pts

project!(at::AbstractAtoms, cons::FixedCell) = at




end
