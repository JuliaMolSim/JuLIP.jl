"""
`module Constraints`

TODO: write documentation
"""
module Constraints

using JuLIP: Dofs, AbstractConstraint, AbstractAtoms,
         mat, vecs, JVecs, JVecsF, JMatF, JMat,
         set_positions!, set_cell!, virial, defm, set_defm!,
         forces, unsafe_positions, momenta, set_momenta!

import JuLIP: position_dofs, project!, set_position_dofs!, positions, gradient, energy,
      momentum_dofs, set_momentum_dofs!


export FixedCell, VariableCell, ExpVariableCell


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


# ========================================================================
#          FIXED CELL IMPLEMENTATION
# ========================================================================

"""
`FixedCell`: the cell shape is fixed; clamp constraints can be placed on
individual atoms, see keyword arguments below.

Momenta for clamped atoms (or atom components) are set to zero.

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
   return sort(find(mask[:]))
end

FixedCell(at::AbstractAtoms; free=nothing, clamp=nothing, mask=nothing) =
   FixedCell(analyze_mask(at, free, clamp, mask))

position_dofs(at::AbstractAtoms, cons::FixedCell) = mat(positions(at))[cons.ifree]

set_position_dofs!(at::AbstractAtoms, cons::FixedCell, x::Dofs) =
      set_positions!(at, positions(at, cons.ifree, x))

momentum_dofs(at::AbstractAtoms, cons::FixedCell) = mat(momenta(at))[cons.ifree]

set_momentum_dofs!(at::AbstractAtoms, cons::FixedCell, p::Dofs) =
      set_momenta!(at, zeros_free(3 * length(at), p, cons.ifree) |> vecs)

project!(at::AbstractAtoms, cons::FixedCell) = at

# TODO: this is a temporaruy hack, and I think we need to
#       figure out how to do this for more general constraints
#       maybe not too terrible
project!(cons::FixedCell, A::SparseMatrixCSC) = A[cons.ifree, cons.ifree]

gradient(at::AbstractAtoms, cons::FixedCell) =
               scale!(mat(forces(at))[cons.ifree], -1.0)

energy(at::AbstractAtoms, cons::FixedCell) = energy(at)


# ========================================================================
#          VARIABLE CELL IMPLEMENTATION
# ========================================================================

"""
`VariableCell`: both atom positions and cell shape are free;

**WARNING:** before manipulating the dof-vectors returned by a `VariableCell`
constraint, read *meaning of dofs* instructions at bottom of help text!

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
`X0, F0`, dofs are understood *relative* to this "initial configuration".

`dofs(at, cons::VariableCell)` returns a vector that represents a pair
`(Y, F1)` of a displacement and a deformation matrix. These are to be understood
*relative* to the reference `X0, F0` stored in `cons` as follows:
* `F = F1`   (the cell is then `F'`)
* `X = [F1 * (F0 \ y)  for y in Y)]`

One aspect of this definition is that clamped atom positions still change via
`F`.
"""
type VariableCell <: AbstractConstraint
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


# ========================================================================
#          EXP VARIABLE CELL IMPLEMENTATION
# ========================================================================
#
# F = exp(U) F0
# x =  F * F0^{-1} z  = exp(U) z
#
# ϕ( exp(U+tV) (z+tv) ) ~ ϕ'(x) ⋅ (exp(U) v) + ϕ'(x) ⋅ ( L(U, V) exp(-U) exp(U) z )
#    >>> ∂E(U) : V  =  [S exp(-U)'] : L(U,V)
#                   =  L'(U, S exp(-U)') : V
#                   =  L(U', S exp(-U)') : V
#                   =  L(U, S exp(-U)) : V     (provided U = U')

import StaticArrays
Base.LinAlg.expm{N,T}(A::StaticArrays.SMatrix{N,N,T}) =
   StaticArrays.SMatrix{N,N,T}(expm(Array(A)))


type ExpVariableCell <: AbstractConstraint
   ifree::Vector{Int}
   X0::JVecsF
   F0::JMatF
   pressure::Float64
   fixvolume::Bool
end


function ExpVariableCell(at::AbstractAtoms;
               free=nothing, clamp=nothing, mask=nothing,
               pressure = 0.0, fixvolume=false)
   if pressure != 0.0 && fixvolume
      warning("the pressure setting will be ignored when `fixvolume==true`")
   end
   return ExpVariableCell( analyze_mask(at, free, clamp, mask),
                        positions(at), defm(at), pressure, fixvolume )
end


function logm_defm(at::AbstractAtoms, cons::ExpVariableCell)
   F = defm(at)
   # remove the reference deformation (F = expm(U) * F0)
   expU = F * inv(cons.F0)
   # expU should be spd, but roundoff (or other things) might mess this up
   # do we need an extra check here?
   U = logm(expU |> Array) |> JMat
   U = real(0.5 * (U + U'))
   # check that expm(U) * F0 ≈ F (if not, then something has gone horribly wrong)
   if vecnorm(F - expm(U) * cons.F0, Inf) > 1e-12
      @show F
      @show expm(U) * cons.F0
      error("something has gone wrong; U is not symmetric?")
   end
   return U, F
end

expposdofs(x) = x[1:end-6]
expcelldofs(x) = x[end-5:end]
U2dofs(U) = U[(1,2,3,5,6,9)]
dofs2U(x) = JMat(expcelldofs(x)[(1,2,3,2,4,5,3,5,6)])

function position_dofs(at::AbstractAtoms, cons::ExpVariableCell)
   X = positions(at)
   U, _ = logm_defm(at, cons)
   # tranform the positions back to the reference cell (F0)
   broadcast!(x -> expm(-U) * x, X, X)
   # construct dof vector
   return [mat(X)[cons.ifree]; U2dofs(U)]
end


function set_position_dofs!(at::AbstractAtoms, cons::ExpVariableCell, x::Dofs)
   U = dofs2U(x)
   expU = expm(U)
   F = expU * cons.F0
   Z = copy(cons.X0)
   mat(Z)[cons.ifree] = expposdofs(x)
   broadcast!(z -> expU * z, Z, Z)
   set_positions!(at, Z)
   set_defm!(at, F)
   return at
end


"""
Directional derivative of matrix exponential. See Theorem 2.1 in
AL-MOHY, HIGHAM SIAM J. Matrix Anal. Appl. 30, 2009.
"""
function dexpm_3x3(X::AbstractMatrix, E::AbstractMatrix)
   @assert size(X) == size(E) == (3, 3)
   return expm([ X E; zeros(3,3) X ])[1:3, 4:6]
end


function gradient(at::AbstractAtoms, cons::ExpVariableCell)
   U, F = logm_defm(at, cons)
   G = forces(at)
   broadcast!(g -> - (expm(U) * g), G, G)   # G[n] <- -exp(U) * G[n]
   T = dexpm_3x3(U, - virial(at) * expm(-U))
   # add the forces acting on the upper diagonal of U to the lower diagonal
   # this way we ensure that T : V = U2dofs(T) : U2dofs(V)
   T[(2,3,6)] += T[(4,7,8)]
   # add the pressure component
   T[(1,5,9)] += cons.pressure * exp(trace(U)) * U[(1,5,9)]
   return [ mat(G)[cons.ifree]; U2dofs(T) ]
end

energy(at::AbstractAtoms, cons::ExpVariableCell) =
               energy(at) + cons.pressure * det(defm(at)) / det(cons.F0)
               #                            >>>>>>  exp(trace(U))  <<<<< (tested)

# TODO: fix this once we implement the volume constraint ??????
project!(at::AbstractAtoms, cons::ExpVariableCell) = at

# TODO: fix the abstraction for projecting a preconditioner;
#       this will actually need to do quite a bit more in the future
# project!(cons::FixedCell, A::SparseMatrixCSC) = A[cons.ifree, cons.ifree]


end # module
