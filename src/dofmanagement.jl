

using JuLIP: Dofs, AbstractConstraint, AbstractAtoms, AbstractCalculator,
             mat, vecs, JMat, JVec,
             set_positions!, set_cell!, virial, cell,
             forces, momenta, set_momenta!,
             constraint, rnn, calculator, hessian_pos

import JuLIP: position_dofs, project!, set_position_dofs!, positions,
              momentum_dofs, set_momentum_dofs!, dofs,
              set_dofs!, positions, energy, gradient, hessian


export FixedCell, VariableCell, InPlaneFixedCell, AntiPlaneFixedCell, atomdofs

using SparseArrays: SparseMatrixCSC, nnz, sparse, findnz

using LinearAlgebra: rmul!, det


struct LinearConstraint{T}
   C::Matrix{T}
   b::Vector{T}
   desc::String   # description of the constraint
end

mutable struct DofManager{T}
   variablecell::Bool
   xfree::Vector{Int}
   lincons::Vector{LinearConstraint{T}}
   # ----
   X0::Vector{JVec{T}}
   F0::JMat{T}
end


# ========================================================================
#          AUXILIARY FUNCTIONS
# ========================================================================



function zeros_free(n::Integer, x::AbstractVector, free)
   z = zeros(eltype(x), n)
   z[free] = x
   return z
end

function insert_free!(p::AbstractArray, x::AbstractVector, free)
   p[free] = x
   return p
end

# a helper function to get a valid positions array from a dof-vector
positions(at::AbstractAtoms,
          ifree::AbstractVector{<:Integer},
          dofs::Dofs) =
   insert_free!(positions(at) |> mat, dofs, ifree) |> vecs

"""
`_pos_to_dof` : a helper function that will convert a positions-based
block-hessian into a classical dof-based hessian with the standard JuLIP
ordering of dofs.
"""
function _pos_to_dof(Hpos::SparseMatrixCSC, at::AbstractAtoms{T}) where {T}
   I, J, Z = Int[], Int[], T[]
   for C in (I, J, Z); sizehint!(C, 9 * nnz(Hpos)); end
   Nat = length(at)
   @assert Nat == size(Hpos, 2)
   # TODO: this findnz creates an extra copy of all data, which we should avoid
   for (iat, jat, zat) in zip(findnz(Hpos)...)
      for a = 1:3, b = 1:3
         push!(I, 3 * (iat-1) + a)
         push!(J, 3 * (jat-1) + b)
         push!(Z, zat[a,b])
      end
   end
   return sparse(I, J, Z, 3*Nat, 3*Nat)
end

"""
`analyze_mask` : helper function to generate list of dof indices from
lists of atom indices indicating free and clamped atoms
"""
function analyze_mask(at, free, clamp, mask)
   if length(findall((free != nothing, clamp != nothing, mask != nothing))) > 1
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
      mask = fill(false, 3, Nat)
      if !isempty(free)
         mask[:, free] .= true
      end
   end
   return findall(mask[:])
end


# ========================================================================
#          MAIN DOF MANAGER INTERFACE
# ========================================================================

"""
`VariableCell`: both atom positions and cell shape are free;

**WARNING:** (1) before manipulating the dof-vectors returned by a `VariableCell`
constraint, read *meaning of dofs* instructions at bottom of help text!

(2) The `volume` field is a signed volume.

Constructor:
```julia
VariableCell(at::AbstractAtoms; free=..., clamp=..., mask=..., fixvolume=false)
```
Set at most one of the kwargs:
* no kwarg: all atoms are free
* `free` : list of free atom indices (not dof indices)
* `clamp` : list of clamped atom indices (not dof indices)
* `mask` : 3 x N Bool array to specify individual coordinates to be clamped

### Meaning of dofs

On call to the constructor, `VariableCell` stored positions and deformation
`X0, F0`, dofs are understood *relative* to this initial configuration.

`dofs(at, cons::VariableCell)` returns a vector that represents a pair
`(Y, F1)` of a displacement and a deformation matrix. These are to be understood
*relative* to the reference `X0, F0` stored in `cons` as follows:
* `F = F1`   (the cell is then `F'`)
* `X = [F1 * (F0 \\ y)  for y in Y)]`

One aspect of this definition is that clamped atom positions still change via
`F`
"""


DofManager(at::AbstractAtoms{T}) where {T} =
      DofManager( false,                    # variablecell
                  collect(1:length(at)),    # xfree
                  LinearConstraint{T}[],    # lincons
                  JVec{T}[],                # X0
                  zero(JMat{T}) )           # F0


set_clamped!(at::Atoms, Iclamp::AbstractVector{<: Integer}) =
   set_mask!(

set_free!(at::Atoms, Ifree::AbstractVector{<: Integer})

set_mask!(at::Atoms, mask::AbstractVector{<: Integer})

               free=nothing, clamp=nothing, mask=nothing,
               pressure = 0.0, fixvolume=false)
   if pressure != 0.0 && fixvolume
      @warn("the pressure parameter will be ignored when `fixvolume==true`")
   end
   return VariableCell( analyze_mask(at, free, clamp, mask),
                        positions(at), cell(at)',
                        pressure, fixvolume, det(cell(at)) )
end
