
module Utils

import JuLIP.Chemistry: rnn

using JuLIP: AbstractAtoms, JVec, JMat, positions, set_positions!,
             chemical_symbols, cell, pbc, mat, dofs, set_dofs!,
             fixedcell, variablecell, apply_defm!

using LinearAlgebra: norm, I

export rattle!, r_sum, r_dot,
       swapxy!, swapxz!, swapyz!,
       dist, displacement, rmin, wrap_pbc!


############################################################
###   Some useful utility functions
############################################################
###   Robust Summation and Dot Products
############################################################

# SWITCH TO KAHAN?
"Robust summation."
r_sum(a) = sum(a)

# NOTE: if I see this correctly, then r_dot allocates a temporary
#       vector, which is likely quite a performance overhead.
#       probably, we want to re-implement this.
"Robust inner product. Defined as `r_dot(a, b) = r_sum(a .* b)`"
r_dot(a, b) = r_sum(a[:] .* b[:])


"""
`rattle!(at::AbstractAtoms, r::Float64; rnn = 1.0, respect_constraint = true)
  -> at`

randomly perturbs the atom positions

* `r`: magnitude of perturbation
* `rnn` : nearest-neighbour distance
* `respect_constraint`: set false to also perturb the constrained atom positions
"""
function rattle!(at::AbstractAtoms{T}, r::AbstractFloat; rnn = 1.0) where {T}
   # TODO: revive the respect_constraint keyword!
     # respect_constraint = (constraint(at) != nothing))
   # if respect_constraint
   #    x = dofs(at)
   #    x += r * rnn * 2.0/sqrt(3) * (rand(Float64, length(x)) .- 0.5)
   #    set_dofs!(at, x)
   # else
   X = positions(at) |> mat
   X .+= r * rnn * 2.0/sqrt(3) * (rand(T, size(X)) .- 0.5)
   set_positions!(at, X)
   if variablecell(at)
      apply_defm!(at, I + (r/rnn) * (rand(JMat{T}) .- 0.5))
   end
   return at
end


"""
swap x and y position coordinates
"""
function swapxy!(at::AbstractAtoms)
   X = positions(at) |> mat
   X[[2,1],:] = X[[1,2],:]
   set_positions!(at, X)
   return at
end

"""
swap x and z position coordinates
"""
function swapxz!(at::AbstractAtoms)
   X = positions(at) |> mat
   X[[3,1],:] = X[[1,3],:]
   set_positions!(at, X)
   return at
end

"""
swap y and z position coordinates
"""
function swapyz!(at::AbstractAtoms)
   X = positions(at) |> mat
   X[[3,2],:] = X[[2,3],:]
   set_positions!(at, X)
   return at
end




"""
* `dist(at, X1, X2, p = Inf)`
* `dist(at1, at2, p = Inf)`
* `dist(at, x1, x2)`

Returns the maximum distance (p = Inf) or alternatively a p-norm of
distances between the two configurations `X1, X2` or `at1, at2`.
This implementation accounts for periodic boundary conditions (in those
coordinate directions where they are set to `true`)
"""
function dist(at::AbstractAtoms,
              X1::AbstractVector{<:JVec}, X2::AbstractVector{<:JVec},
              p = Inf)
   @assert length(X1) == length(X2) == length(at)
   F = cell(at)'
   Finv = inv(F)
   d = [ pernorm(F, Finv, pbc(at), x1 - x2) for (x1, x2) in zip(X1, X2) ]
   return norm(d, p)
end

function dist(at::AbstractAtoms, x1::JVec, x2::JVec)
   F = cell(at)'
   Finv = inv(F)
   return pernorm(F, Finv, pbc(at), x1 - x2)
end

pernorm(F, Finv, p, x) = norm(_project_pbc_(F, Finv, p, x))

dist(at::AbstractAtoms, X::AbstractVector) = dist(at, positions(at), X)

function dist(at1::AbstractAtoms, at2::AbstractAtoms, p = Inf)
   @assert norm(cell(at1) - cell(at2), Inf) < 1e-14
   return dist(at1, positions(at1), positions(at2), p)
end

_myrem(λ::Real, p::Bool) = p ? rem(λ, 1.0, RoundNearest) : λ

function _project_pbc_(F, Finv, p, x)
   λ = Finv * x     # convex coordinates
   # convex coords projected to the unit
   λp = JVec(_myrem(λ[1], p[1]), _myrem(λ[2], p[2]), _myrem(λ[3], p[3]))
   return F * λp     # convert back to real coordinates
end


function _project_coord_min_(λ, p)
   if !p
      return λ
   end
   λ = mod(λ, 1.0)   # project to cell
   if λ > 0.5        # periodic image with minimal length
      λ = λ - 1.0
   end
   return λ
end

function _project_pbc_min_(F, Finv, p, x)
   λ = Finv * x     # convex coordinates
   # convex coords projected to the unit
   λp = _project_coord_min_.(λ, JVec{Bool}(p...))
   return F * λp     # convert back to real coordinates
end


function displacement(at::AbstractAtoms,
                      X1::AbstractVector{<:JVec},
                      X2::AbstractVector{<:JVec})
   @assert length(X1) == length(X2) == length(at)
   F = cell(at)'
   Finv = inv(F)
   p = pbc(at)
   U = [ _project_pbc_min_(F, Finv, p, x2-x1)
         for (x1, x2) in zip(X1, X2) ]
   return U
end

project_min(at, u) = _project_pbc_min_(cell(at)', inv(cell(at)'), pbc(at), u)

function rnn(at::AbstractAtoms)
   syms = unique(chemical_symbols(at))
   return minimum(rnn.(syms))
end

function wrap_pbc!(at)
   X = positions(at)
   F = cell(at)'
   X = [_project_pbc_min_(F, inv(F), pbc(at), x) for x in X]
   set_positions!(at, X)
end


function rmin(at::AbstractAtoms)
   at2 = at * 2
   X = positions(at2)
   r = norm(X[1]-X[2])
   for n = 1:length(X)-1, m = n+1:length(X)
      r = min(r, norm(X[n]-X[m]))
   end
   return r
end

end
