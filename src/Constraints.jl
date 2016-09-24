
"""
`module Constraints`

TODO: write documentation
"""
module Constraints

using JuLIP: Dofs, AbstractConstraint, mat, vecs, JVecs, AbstractAtoms,
         set_positions!

import JuLIP: dofs, project!, set_dofs!, positions, gradient


export FixedCell, VariableCell


function zeros_free{T}(n::Integer, x::Vector{T}, free::Vector{Int})
   z = zeros(T, n)
   z[free] = x
   return z
end

function insert_free!{T}(p::Array{T}, x::Vector{T}, free::Vector{Int})
   p[free] = x
   return p
end

# a helper function to get a valid positions array from a dof-vector
positions{TI<:Integer}(at::AbstractAtoms, ifree::AbstractVector{TI}, dofs::Dofs) =
      insert_free!(positions(at) |> mat, dofs, ifree) |> vecs



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
* `mask` : 3 x N Bool array to specify individual coordinates to be clamped
"""
type FixedCell <: AbstractConstraint
   ifree::Vector{Int}
end

function analyze_mask(at, free, clamp, mask)
   if length(find((free != nothing, clamp != nothing, mask != nothing))) > 1
      error("FixedCell: only one of `free`, `clamp`, `mask` may be provided")
   elseif all( (free == nothing, clamp == nothing, mask == nothing) )
      # in this case (default) all atoms are free
      return collect(1:3*length(at))
   end
   # determine free dof indices
   Nat = length(at)
   if clamp != nothing
      # revert to setting free
      free = setdiff(1:Nat, clamp)
   end
   if free != nothing
      # revert to setting mask
      mask = Matrix{Bool}(3, Nat)
      fill!(mask, false)
      mask[:, free] = true
   end
   return mask[:]
end

FixedCell(at::AbstractAtoms; free=nothing, clamp=nothing, mask=nothing) =
   FixedCell(analyze_mask(at, free, clamp, mask))

# convert positions to a dofs vector; TODO: use unsafe_positions????
dofs(at::AbstractAtoms, cons::FixedCell) = mat(positions(at))[cons.ifree]

set_dofs!(at::AbstractAtoms, cons::FixedCell, x::Dofs) =
      set_positions!(at, positions(at, cons.ifree, x))

project!(at::AbstractAtoms, cons::FixedCell) = at

# TODO: this is a temporaruy hack, and I think we need to
#       figure out how to do this for more general constraints
#       maybe not too terrible
project!(cons::FixedCell, A::SparseMatrixCSC) = A[cons.ifree, cons.ifree]

gradient(at::AbstractAtoms, cons::FixedCell) = mat(gradient(at))[cons.ifree]



"""
`VariableCell`: both atom positions and cell shape are free

Constructor:
```julia
VariableCell(at::AbstractAtoms; free=..., clamp=..., mask=..., fixvolume=false)
```
Set at most one of the kwargs:
* no kwarg: all atoms are free
* `free` : list of free atom indices (not dof indices)
* `clamp` : list of clamped atom indices (not dof indices)
* `mask` : 3 x N Bool array to specify individual coordinates to be clamped
* `fixvolume` : {false}; set true if the cell volume should be fixed
"""
type VariableCell <: AbstractConstraint
   ifree::Vector{Int}
   fixvolume::Bool
end

VariableCell(at::AbstractAtoms;
               free=nothing, clamp=nothing, mask=nothing,
               fixvolume=false) =
   VariableCell(analyze_mask(at, free, clamp, mask), fixvolume)

# F = Q U, U spd but in any case symmetric, so we should just allow F
# to be symmetric.
#
# consider the perturbation
#  F   -> F + t U
#  x_i -> (F+t U) F^{-1} x_i = x_i + t U F^{-1} x_i  =: x_i^t
# E({x_i^t}) ~ E({x_i}) + t ∑_i g_i ⋅ (U F^{-1} x_i) + O(t^2)
#            ~ E({x_i}) + t ∑_i g_ia U_ab [F_{-1} x_i]_b
#            ~ E({x_i}) + t U_ab : [ ∑_i g_{ia} [F^{-1} x_i]_b  ]_ab
#            ~ E({x_i}) + t U : ∑_i g_i ⊗ (F^{-1} x_i)
#
# now consider
#  x_i^t = (F + t U) F^{-1} (x_i + t u_i)
#        ~ x_i + t u_i + t U F^{-1} x_i + O(t^2)
#
#   small issue: this is a nonlinear search path !!!

dofs(at::AbstractAtoms, cons::VariableCell) =
         [ mat(positions(at))[cons.ifree]; cell(at)[:] ]

celldofs(x) = x[end-8:end]

function set_dofs!(at::AbstractAtoms, cons::VariableCell, x::Dofs)
   set_positions!(at, positions(at, cons.ifree, x))
   set_cell!(at, celldofs(x))
end

function gradient(at::AbstractAtoms, cons::VariableCell)
   G = gradient(at)   # - forces ( = gradient assuming cell is fixed)
   X = positions(at)
   Finv = cell(at) |> JMat |> inv
   S = sum( g * (Finv * x)'  for (g, x) in zip(G, X) )
   return [ mat(G)[cons.ifree]; Array(S[:]) ]
end

# TODO: fix this once we implement the volume constraint ??????
project!(at::AbstractAtoms, cons::VariableCell) = at

# project!(cons::FixedCell, A::SparseMatrixCSC) = A[cons.ifree, cons.ifree]


end # module
