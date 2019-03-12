
"""
`module Constraints`

TODO: write documentation
"""
module Constraints

using JuLIP: Dofs, AbstractConstraint, AbstractAtoms,
             mat, vecs, JVecs, JVecsF, JMatF, JMat,
             set_positions!, set_cell!, virial, defm, set_defm!,
             forces, momenta, set_momenta!,
             constraint, rnn, AbstractCalculator, calculator

# temporary hack - we may need to rethink this
using JuLIP.Potentials: hessian_pos

import JuLIP: position_dofs, project!, set_position_dofs!, positions, gradient,
              energy, momentum_dofs, set_momentum_dofs!, dofs, project!,
              set_dofs!, positions, gradient, energy, hessian


export FixedCell, VariableCell, ExpVariableCell, FixedCell2D, atomdofs

using SparseArrays: SparseMatrixCSC, nnz, sparse, findnz

using LinearAlgebra: rmul!, det


function zeros_free(n::Integer, x::Vector{T}, free::Vector{Int}) where T
   z = zeros(T, n)
   z[free] = x
   return z
end

function insert_free!(p::AbstractArray{T}, x::AbstractVector{T},
                      free::AbstractVector{Int}) where T
   p[free] = x
   return p
end

# a helper function to get a valid positions array from a dof-vector
positions(at::AbstractAtoms, ifree::AbstractVector{TI}, dofs::Dofs) where {TI<:Integer} =
      insert_free!(positions(at) |> mat, dofs, ifree) |> vecs


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
mutable struct FixedCell <: AbstractConstraint
   ifree::Vector{Int}
end

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
      mask = Matrix{Bool}(undef, 3, Nat)
      fill!(mask, false)
      if !isempty(free)
         mask[:, free] = true
      end
   end
   return findall(mask[:])
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

# TODO: this is a temporary hack, and I think we need to
#       figure out how to do this for more general constraints
#       maybe not too terrible
#       but why is there a !???
project!(cons::FixedCell, A::SparseMatrixCSC) = A[cons.ifree, cons.ifree]

gradient(at::AbstractAtoms, cons::FixedCell) =
               rmul!(mat(forces(at))[cons.ifree], -1.0)

energy(at::AbstractAtoms, cons::FixedCell) = energy(at)

# >>> TODO: move to abstractions.jl ??
hessian(at::AbstractAtoms, cons::FixedCell) =
      project!(cons, _pos_to_alldof(hessian_pos(calculator(at), at), at))

function _pos_to_alldof(Hpos::SparseMatrixCSC, at::AbstractAtoms)
   I, J, Z = Int[], Int[], Float64[]
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


# ===================
#   In-Plane Motion
# ===================


FixedCell2D(at::AbstractAtoms; kwargs...) = InPlaneFixedCell(at; kwargs...)

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

# =====================
#   Anti-Plane Motion
# =====================

"""
preliminary implementation of a Constraint, restricting a
simulation to 2D in-plane motion
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

# """
# `VariableCell`: both atom positions and cell shape are free;
#
# **WARNING:** before manipulating the dof-vectors returned by a `VariableCell`
# constraint, read *meaning of dofs* instructions at bottom of help text!
#
# Constructor:
# ```julia
# VariableCell(at::AbstractAtoms; free=..., clamp=..., mask=..., fixvolume=false)
# ```
# Set at most one of the kwargs:
# * no kwarg: all atoms are free
# * `free` : list of free atom indices (not dof indices)
# * `clamp` : list of clamped atom indices (not dof indices)
# * `mask` : 3 x N Bool array to specify individual coordinates to be clamped
#
# ### Meaning of dofs
#
# On call to the constructor, `VariableCell` stored positions and deformation
# `X0, F0`, dofs are understood *relative* to this "initial configuration".
#
# `dofs(at, cons::VariableCell)` returns a vector that represents a pair
# `(Y, F1)` of a displacement and a deformation matrix. These are to be understood
# *relative* to the reference `X0, F0` stored in `cons` as follows:
# * `F = F1`   (the cell is then `F'`)
# * `X = [F1 * (F0 \ y)  for y in Y)]`
#
# One aspect of this definition is that clamped atom positions still change via
# `F`.
# """
mutable struct VariableCell <: AbstractConstraint
   ifree::Vector{Int}
   X0::JVecsF
   F0::JMatF
   pressure::Float64
   fixvolume::Bool
   volume::Float64    # this is meaningless if `fixvolume == false`
end


function VariableCell(at::AbstractAtoms;
               free=nothing, clamp=nothing, mask=nothing,
               pressure = 0.0, fixvolume=false)
   if pressure != 0.0 && fixvolume
      warning("the pressure setting will be ignores when `fixvolume==true`")
   end
   return VariableCell( analyze_mask(at, free, clamp, mask),
                        positions(at), defm(at),
                        pressure, fixvolume, det(defm(at)) )
end

# reverse map:
#   F -> F
#   X[n] = F * F^{-1} X0[n]

function position_dofs(at::AbstractAtoms, cons::VariableCell)
   X = positions(at)
   F = defm(at)
   A = cons.F0 * inv(F)
   U = [A * x for x in X]   # switch to broadcast!
   return [mat(U)[cons.ifree]; Matrix(F)[:]]
end


posdofs(x) = x[1:end-9]
celldofs(x) = x[end-8:end]

function set_position_dofs!(at::AbstractAtoms, cons::VariableCell, x::Dofs)
   F = JMatF(celldofs(x))
   A = F * inv(cons.F0)
   Y = copy(cons.X0)
   mat(Y)[cons.ifree] = posdofs(x)
   for n = 1:length(Y)
      Y[n] = A * Y[n]
   end
   set_positions!(at, Y)
   set_defm!(at, F)
   return at
end

# for a variation x^t_i = (F+tU) F_0^{-1} (u_i + t v_i)
#       ~ U F^{-1} F F0^{-1} u_i + F F0^{-1} v_i
# we get
#      dE/dt |_{t=0} = U : (S F^{-T}) - < (F * inv(F0))' * frc, v>
#
# this is nice because there is no contribution from the stress to
# the positions component of the gradient

vol(at::AbstractAtoms) = det(defm(at))

vol_d(at::AbstractAtoms) = vol(at) * inv(defm(at))'

# function vol_dd(at::AbstractAtoms)
#    hdetI = zeros(3,3,3,3)
#    h = 0.1
#    for i = 1:3, j = 1:3
#       Ih = eye(3); Ih[i,j] += h
#       hdetI[:,:,i,j] = (ddet(Ih) - ddetI) / h
#    end
#    round(Int, reshape(hdetI, 9, 9))
# end

function gradient(at::AbstractAtoms, cons::VariableCell)
   F = defm(at)
   A = F * inv(cons.F0)
   G = forces(at)
   for n = 1:length(G)
      G[n] = - A' * G[n]
   end
   S = - virial(at) * inv(F)'        # ∂E / ∂F
   S += cons.pressure * vol_d(at)     # applied stress
   return [ mat(G)[cons.ifree]; Array(S)[:] ]
end

energy(at::AbstractAtoms, cons::VariableCell) =
         energy(at) + cons.pressure * det(defm(at))

# TODO: fix this once we implement the volume constraint ??????
project!(at::AbstractAtoms, cons::VariableCell) = at

# TODO: fix the abstraction for projecting a preconditioner;
#       this will actually need to do quite a bit more in the future
# project!(cons::FixedCell, A::SparseMatrixCSC) = A[cons.ifree, cons.ifree]

# TODO: CONTINUE WITH EXPCELL IMPLEMENTATION
# include("expcell.jl")

# convenience function to return DoFs associated with a particular atom
atomdofs(a::AbstractAtoms, I::Integer) = findall(in(3*I-2:3*I), constraint(a).ifree)

end # module
