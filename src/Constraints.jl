# TODO: switch all cell(at)' to cell(at) and store cells


"""
`module Constraints`

TODO: write documentation
"""
module Constraints

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







# ========================================================================
#          FIXED CELL IMPLEMENTATION
# ========================================================================

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
struct FixedCell <: AbstractConstraint
   ifree::Vector{Int}
end

FixedCell(at::AbstractAtoms; clamp = nothing, free=nothing, mask=nothing) =
   FixedCell(analyze_mask(at, free, clamp, mask))

position_dofs(at::AbstractAtoms, cons::FixedCell) = mat(positions(at))[cons.ifree]

set_position_dofs!(at::AbstractAtoms, cons::FixedCell, x::Dofs) =
      set_positions!(at, positions(at, cons.ifree, x))

momentum_dofs(at::AbstractAtoms, cons::FixedCell) = mat(momenta(at))[cons.ifree]

set_momentum_dofs!(at::AbstractAtoms, cons::FixedCell, p::Dofs) =
      set_momenta!(at, zeros_free(3 * length(at), p, cons.ifree) |> vecs)

project!(at::AbstractAtoms, cons::FixedCell) = at

gradient(calc::AbstractCalculator, at::AbstractAtoms, cons::FixedCell) =
      rmul!(mat(forces(at))[cons.ifree], -1.0)

energy(calc::AbstractCalculator, at::AbstractAtoms, cons::FixedCell) =
      energy(calc, at)


# TODO: hacking the hessian dispatch until we can figure out a nicer way to
# achieve this?

hessian(calc::AbstractCalculator, at::AbstractAtoms, cons::FixedCell) =
      project(cons, _pos_to_dof(hessian_pos(calc, at), at))

# TODO: this is a temporary hack, and I think we need to
#       figure out how to do this for more general constraints
#       maybe not too terrible
project(cons::FixedCell, A::SparseMatrixCSC) = A[cons.ifree, cons.ifree]

# ===========================
#   2D FixedCell Constraints
# ===========================

"""
preliminary implementation of a Constraint, restricting a
simulation to 2D in-plane motion
"""
function InPlaneFixedCell(at::AbstractAtoms; clamp = Int[], free = nothing)
   if free == nothing
      free = setdiff(1:length(at), clamp)
   end
   mask = fill(false, (3, length(at)))
   mask[:, free] = true
   mask[3, :] = false
   return FixedCell(at, mask = mask)
end

"""
preliminary implementation of a Constraint, restricting a
simulation to 2D out-of-plane motion
"""
function AntiPlaneFixedCell(at::AbstractAtoms; free = Int[])
   mask = fill(false, (3, length(at)))
   mask[:, free] = true
   mask[1:2, :] = false
   return FixedCell(at, mask = mask)
end



# ========================================================================
#          VARIABLE CELL IMPLEMENTATION
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
struct VariableCell{T} <: AbstractConstraint
   ifree::Vector{Int}
   X0::Vector{JVec{T}}
   F0::JMat{T}
   pressure::T
   fixvolume::Bool
   volume::T    # this is meaningless if `fixvolume == false`
end


function VariableCell(at::AbstractAtoms;
               free=nothing, clamp=nothing, mask=nothing,
               pressure = 0.0, fixvolume=false)
   if pressure != 0.0 && fixvolume
      @warn("the pressure parameter will be ignored when `fixvolume==true`")
   end
   return VariableCell( analyze_mask(at, free, clamp, mask),
                        positions(at), cell(at)',
                        pressure, fixvolume, det(cell(at)) )
end

# reverse map:
#   F -> F
#   X[n] = F * F^{-1} X0[n]

function position_dofs(at::AbstractAtoms, cons::VariableCell)
   X = positions(at)
   F = cell(at)'
   A = cons.F0 * inv(F)
   U = [A * x for x in X]   # switch to broadcast!
   return [mat(U)[cons.ifree]; Matrix(F)[:]]
end

posdofs(x) = x[1:end-9]
celldofs(x) = x[end-8:end]

function set_position_dofs!(at::AbstractAtoms{T}, cons::VariableCell, x::Dofs
                            ) where {T}
   F = JMat{T}(celldofs(x))
   A = F * inv(cons.F0)
   Y = copy(cons.X0)
   mat(Y)[cons.ifree] = posdofs(x)
   for n = 1:length(Y)
      Y[n] = A * Y[n]
   end
   set_positions!(at, Y)
   set_cell!(at, F')
   return at
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
# sigvol_d(at::AbstractAtoms) = sigvol(at) * inv(cell(at))

function gradient(calc::AbstractCalculator, at::AbstractAtoms, cons::VariableCell)
   F = cell(at)'
   A = F * inv(cons.F0)
   G = forces(at)
   for n = 1:length(G)
      G[n] = - A' * G[n]
   end
   S = - virial(at) * inv(F)'             # ∂E / ∂F
   S += cons.pressure * sigvol_d(at)'     # applied stress
   return [ mat(G)[cons.ifree]; Array(S)[:] ]
end


energy(calc::AbstractCalculator, at::AbstractAtoms, cons::VariableCell) =
         energy(calc, at) + cons.pressure * sigvol(at)

# TODO: fix this once we implement the volume constraint ??????
#       => or just disallow the volume constraint for now?
project!(at::AbstractAtoms, cons::VariableCell) = at


# TODO: CONTINUE WITH EXPCELL IMPLEMENTATION
# include("expcell.jl")

# convenience function to return DoFs associated with a particular atom
atomdofs(a::AbstractAtoms, I::Integer) = findall(in(3*I-2:3*I), constraint(a).ifree)

end # module
