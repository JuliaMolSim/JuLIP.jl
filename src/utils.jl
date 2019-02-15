
import JuLIP.Chemistry: rnn

export rattle!, r_sum, r_dot,
      swapxy!, swapxz!, swapyz!,
      dist, displacement, rmin


############################################################
###   Some useful utility functions
############################################################
###   Robust Summation and Dot Products
############################################################

"Robust summation."
r_sum(a) = sum(a)


## NOTE: if I see this correctly, then r_dot allocates a temporary
##       vector, which is likely quite a performance overhead.
##       probably, we want to re-implement this.
## TODO: new version without the intermediate allocation
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
function rattle!(at::AbstractAtoms, r::Float64; rnn = 1.0, respect_constraint = true)
   # if there is no constraint, then revert to respect_constraint = false
   if isa(constraint(at), NullConstraint)
      respect_constraint = false
   end
   if respect_constraint
      x = dofs(at)
      x += r * rnn * 2.0/sqrt(3) * (rand(Float64, length(x)) .- 0.5)
      set_dofs!(at, x)
   else
      X = positions(at) |> mat
      X += r * rnn * 2.0/sqrt(3) * (rand(Float64, size(X)) .- 0.5)
      set_positions!(at, X)
   end
   return at
end


"""
use this instead of `warn`, then warnings can be turned off by setting
`Main.JULIPWARN=false`
"""
function julipwarn(s)
   if isdefined(Main, :JULIPWARN)
      if Main.JULIPWARN == false
         return false
      end
   end
   warn(s)
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
`dist(at, X1, X2, p = Inf)`
`dist(at1, at2, p = Inf)`

Returns the maximum distance (p = Inf) or alternatively a p-norm of
distances between the two configurations `X1, X2` or `at1, at2`.
This implementation accounts for periodic boundary conditions (in those
coordinate directions where they are set to `true`)
"""
function dist(at::AbstractAtoms,
              X1::AbstractVector{JVec{T}}, X2::AbstractVector{JVec{T}},
              p = Inf) where T
   @assert length(X1) == length(X2)
   F = cell(at)'
   Finv = inv(F)
   bcrem = [ p ? 1.0 : Inf for p in pbc(at) ]
   d = [ norm(_project_pbc_(F, Finv, bcrem, x1 - x2))
         for (x1, x2) in zip(X1, X2) ]
   return norm(d, p)
end

dist(at::AbstractAtoms, X::AbstractVector) = dist(at, positions(at), X)

function dist(at1::AbstractAtoms, at2::AbstractAtoms, p = Inf)
   @assert norm(cell(at1) - cell(at2), Inf) < 1e-14
   return dist(at1, positions(at1), positions(at2), p)
end

function _project_pbc_(F, Finv, bcrem, x)
   λ = Finv * x     # convex coordinates
   # convex coords projected to the unit
   λp = JVecF(rem(λ[1], bcrem[1]), rem(λ[2], bcrem[2]), rem(λ[3], bcrem[3]))
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
                      X1::AbstractVector{JVec{T}},
                      X2::AbstractVector{JVec{T}}
                ) where T
   @assert length(X1) == length(X2)
   F = defm(at)
   Finv = inv(F)
   p = pbc(at)
   U = [ _project_pbc_min_(F, Finv, p, x2-x1)
         for (x1, x2) in zip(X1, X2) ]
   return U
end

project_min(at, u) = _project_pbc_min_(defm(at), inv(defm(at)), pbc(at), u)

function rnn(at::AbstractAtoms)
   syms = unique(chemical_symbols(at))
   return minimum(rnn.(syms))
end

function wrap_pbc!(at)
   X = positions(at)
   F = defm(at)
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
