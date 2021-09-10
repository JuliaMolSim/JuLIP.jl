

export atomdofs, variablecell, fixedcell, dofmgr_resetref!, variablecell!,
       fixedcell!, set_clamp!, set_free!, set_mask!, reset_clamp!,
       inplane!

using SparseArrays: SparseMatrixCSC, nnz, sparse, findnz

using LinearAlgebra: rmul!, det



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

# TODO: this is a temporary hack, and I think we need to
#       figure out how to do this for more general constraints
#       maybe not too terrible
projectxfree(at::Atoms, A::SparseMatrixCSC) =
      A[at.dofmgr.xfree, at.dofmgr.xfree]

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

analyze_mask(at, free::Colon, clamp::Nothing, mask::Nothing) =
      collect(1:3*length(at))

analyze_mask(at, free::Nothing, clamp::Colon, mask::Nothing) =
      Int[]


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


DofManager(at::AbstractAtoms) = DofManager(length(at), eltype(at))

DofManager(Nat::Integer, T = Float64) =
      DofManager( false,                      # variablecell
                  collect(1:3*Nat),    # xfree
                  LinearConstraint{T}[],      # lincons
                  JVec{T}[],                  # X0
                  zero(JMat{T}) )             # F0

import Base.==
==(d1::DofManager, d2::DofManager) =
      (d1.variablecell == d2.variablecell) && (d1.xfree == d2.xfree)
      # TODO: potential need to add equality about the constraints?
      #       is xfree even relevant for equality?

variablecell(at::Atoms) = at.dofmgr.variablecell
fixedcell(at::Atoms) = !variablecell(at::Atoms)

function dofmgr_resetref!(at)
   at.dofmgr.X0 = copy(positions(at))
   at.dofmgr.F0 = cell(at)'
   return at
end


# convenience function to return DoFs associated with a particular atom
atomdofs(at::Atoms, i::Integer) =
      findall(in(3*i-2:3*i), at.dofmgr.xfree)


"""
`variablecell!(at)` : make the cell-shape variable
"""
variablecell!(at) = set_variablecell!(at, true)

"""
`fixedcell!(at)` : freeze the cell shape
"""
fixedcell!(at) = set_variablecell!(at, false)

"""
`set_variablecell!` : specify whether cell shape is variable or fixed
"""
function set_variablecell!(at::AbstractAtoms, tf::Bool)
   at.dofmgr.variablecell = tf
   if tf
      dofmgr_resetref!(at)
   end
   return at
end


set_clamp!(at::Atoms, Iclamp::AbstractVector{<: Integer}) =
      set_clamp!(at; clamp = Iclamp)

set_free!(at::Atoms, Ifree::AbstractVector{<: Integer}) =
      set_clamp!(at; free = Ifree)

set_mask!(at::Atoms, mask) =
      set_clamp!(at; mask=mask)

reset_clamp!(at) = set_clamp!(at; free = :)

function set_clamp!(at::Atoms; free=nothing, clamp=nothing, mask=nothing)
   at.dofmgr.xfree = analyze_mask(at, free, clamp, mask)
   return at
end


# reverse map:
#   F -> F
#   X[n] = F * F^{-1} X0[n]

function position_dofs(at::Atoms)
   if fixedcell(at)
      return mat(at.X)[at.dofmgr.xfree]
   end
   # variable cell case:
   X = positions(at)
   F = cell(at)'
   A = at.dofmgr.F0 * inv(F)
   U = [A * x for x in X]   # switch to broadcast!
   return [mat(U)[at.dofmgr.xfree]; Matrix(F)[:]]
end

posdofs(x) = x[1:end-9]

celldofs(x) = x[end-8:end]

function set_position_dofs!(at::AbstractAtoms{T}, x::Dofs) where {T}
   if fixedcell(at)
      return set_positions!(at, positions(at, at.dofmgr.xfree, x))
   end
   # variable cell case:
   F = JMat{T}(celldofs(x))
   A = F * inv(at.dofmgr.F0)
   Y = copy(at.dofmgr.X0)
   mat(Y)[at.dofmgr.xfree] = posdofs(x)
   for n = 1:length(Y)
      Y[n] = A * Y[n]
   end
   set_positions!(at, Y)
   set_cell!(at, F')
   return at
end

function momentum_dofs(at::AbstractAtoms)
   if fixedcell(at)
      return mat(momenta(at))[at.dofmgr.xfree]
   end
   @error("`momentum_dofs` is not yet implmement for variable cells")
   return nothing
end

function set_momentum_dofs!(at::AbstractAtoms, p::Dofs)
   if fixedcell(at)
      return set_momenta!(at, zeros_free(3 * length(at), p, at.dofmgr.xfree) |> vecs)
   end
   @error("`set_momentum_dofs!` is not yet implmement for variable cells")
   return nothing
end


# for a variation x^t_i = (F+tU) F_0^{-1} (u_i + t v_i)
#       ~ U F^{-1} F F0^{-1} u_i + F F0^{-1} v_i
# we get
#      dE/dt |_{t=0} = U : (S F^{-T}) - < (F * inv(F0))' * frc, v>
#
# this is nice because there is no contribution from the stress to
# the positions component of the gradient

"""
`sigvol` : signed volume
"""
sigvol(C::AbstractMatrix) = det(C)
sigvol(at::AbstractAtoms) = sigvol(cell(at))

"""
`sigvol_d` : derivative of signed volume
"""
sigvol_d(C::AbstractMatrix) = sigvol(C) * inv(C)'
sigvol_d(at::AbstractAtoms) = sigvol_d(cell(at))

function gradient(calc::AbstractCalculator, at::Atoms)
   if fixedcell(at)
      return rmul!(mat(forces(calc, at))[at.dofmgr.xfree], -1.0)
   end
   F = cell(at)'
   A = F * inv(at.dofmgr.F0)
   G = forces(calc, at)
   for n = 1:length(G)
      G[n] = - A' * G[n]
   end
   S = - virial(calc, at) * inv(F)'                  # ∂E / ∂F
   # S += at.dofmgr.pressure * sigvol_d(at)'     # applied stress  TODO: revive this!
   return [ mat(G)[at.dofmgr.xfree]; Array(S)[:] ]
end

function hessian(calc::AbstractCalculator, at::AbstractAtoms)
   if fixedcell(at)
      H =  _pos_to_dof(hessian_pos(calc, at), at)
      return projectxfree(at, H)
   end
   @error("`hessian` is not yet implmement for variable cells")
   return nothing
end


function inplane!(at::Atoms; free = 1:length(at), clamp = nothing, i1 = 1, i2 = 2)
   if clamp != nothing
      free = setdiff(1:length(at), clamp)
   end
   mask = fill(false, 3, length(at))
   mask[i1, free] .= true
   mask[i2, free] .= true
   set_mask!(at, mask)
   return at
end


# TODO:
#   - anti-plane
#   - pressure
#   - fixvolume
#   - once we add an external potential we need to think about terminology
#     should energy -> potential_energy?; total_energy a new one?
#     total_energy(calc::AbstractCalculator, at::AbstractAtoms, cons::VariableCell) =
#              energy(calc, at) + cons.pressure * sigvol(at)
