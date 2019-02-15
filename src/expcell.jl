# ========================================================================
#          EXP VARIABLE CELL IMPLEMENTATION
# ========================================================================
#
# F = exp(U) F0
# x =  F * F0^{-1} z  = exp(U) z
#
# ϕ( exp(U+tV) (z+tv) ) ~ ϕ'(x) ⋅ (exp(U) V) + ϕ'(x) ⋅ ( L(U, V) exp(-U) exp(U) z )
#    >>> ∂E(U) : V  =  [S exp(-U)'] : L(U,V)
#                   =  L'(U, S exp(-U)') : V
#                   =  L(U', S exp(-U)') : V
#                   =  L(U, S exp(-U)) : V     (provided U = U')



# define expm for StaticArrays since it is not provided (anymre?)

import StaticArrays

using LinearAlgebra: det

_expm(A::StaticArrays.SMatrix{N,N,T}) where {N,T} =
   StaticArrays.SMatrix{N,N,T}(exp(Array(A)))

mutable struct ExpVariableCell <: AbstractConstraint
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
   U = log(expU |> Array) |> JMat
   U = real(0.5 * (U + U'))
   # check that expm(U) * F0 ≈ F (if not, then something has gone horribly wrong)
   if norm(F - _expm(U) * cons.F0, Inf) > 1e-12
      @show F
      @show _expm(U) * cons.F0
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
   broadcast!(x -> _expm(-U) * x, X, X)
   # construct dof vector
   return [mat(X)[cons.ifree]; U2dofs(U)]
end


function set_position_dofs!(at::AbstractAtoms, cons::ExpVariableCell, x::Dofs)
   U = dofs2U(x)
   expU = _expm(U)
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
   return _expm([ X E; zeros(3,3) X ])[1:3, 4:6]
end


function gradient(at::AbstractAtoms, cons::ExpVariableCell)
   U, F = logm_defm(at, cons)
   G = forces(at)
   broadcast!(g -> - (_expm(U) * g), G, G)   # G[n] <- -exp(U) * G[n]
   T = dexpm_3x3(U, - virial(at) * _expm(-U))
   # add the forces acting on the upper diagonal of U to the lower diagonal
   # this way we ensure that T : V = U2dofs(T) : U2dofs(V)
   T[[2,3,6]] += T[[4,7,8]]
   # add the pressure component
   T[[1,5,9]] += cons.pressure * exp(trace(U)) * U[[1,5,9]]
   return [ mat(G)[cons.ifree]; U2dofs(T) ]
end

energy(at::AbstractAtoms, cons::ExpVariableCell) =
               energy(at) + cons.pressure * det(defm(at)) / det(cons.F0)
               #                            >>>>>>  exp(trace(U))  <<<<< (tested)

# TODO: fix this once we implement the volume constraint ??????
project!(at::AbstractAtoms, cons::ExpVariableCell) = at
