
module Preconditioners

using JuLIP: AbstractAtoms, Preconditioner, JVecs, JVecsF, Dofs, maxdist,
            constraint, pairs, cutoff, positions, defm, JVecF, forces, mat,
            set_positions!, julipwarn, chemical_symbols, rnn
using JuLIP.Potentials: @analytic, evaluate, PairPotential, HS
using JuLIP.Constraints: project!, FixedCell

try
   using PyAMG: RugeStubenSolver
catch
   julipwarn("failed to load `PyAMG`")
end

import JuLIP.Potentials: precon

import JuLIP: update!
import Base: A_ldiv_B!, A_mul_B!, dot, *, \

export Exp, FF


# ================ AMGPrecon =====================
# this is wrapping some machinery around PyAMG

# TODO: allow direct solver as an alternative;
#       in this case we should to a cholesky factorisation
#       on `force_update!`

abstract type PairPrecon <: Preconditioner end

"""
`AMGPrecon{T}`: a preconditioner using AMG as the main solver

`AMGPrecon` stores a field `p` which is used to determine the preconditioner
matrix via
```julia
precon_matrix(P, at::AbstractAtoms)
```
If `p` should be updated in response to an update of `at` then
the type can overload `JuLIP.Preconditioners.update_inner!`.

## Constructor:
```
AMGPrecon(p::Any, at::AbstractAtoms; updatedist=0.3, tol=1e-7)
```
## TODO:
* provide mechanism for automatically determining updatedist
from the nearest-neighbour distance in `at`
* should also have automatic updates every 10 or so iterations
"""
type AMGPrecon{T} <: PairPrecon
   p::T
   amg::RugeStubenSolver
   oldX::JVecsF
   updatedist::Float64
   tol::Float64
   updatefreq::Int
   skippedupdates::Int
   stab::Float64
end


type DirectPrecon{T} <: PairPrecon
   p::T
   A::SparseMatrixCSC
   oldX::JVecsF
   updatedist::Float64
   tol::Float64
   updatefreq::Int
   skippedupdates::Int
   stab::Float64
end

function AMGPrecon(p, at::AbstractAtoms;
         updatedist=0.3, tol=1e-7, updatefreq=10, stab=0.01)
   # make sure we don't use this in a context it is not intended for!
   @assert isa(constraint(at), FixedCell)
   P = AMGPrecon(p, RugeStubenSolver(speye(2)), copy(positions(at)),
                     updatedist, tol, updatefreq, 0, stab)
   return force_update!(P, at)
end

function DirectPrecon(p, at::AbstractAtoms;
         updatedist=0.3, tol=1e-7, updatefreq=10, stab=0.01)
   # make sure we don't use this in a context it is not intended for!
   @assert isa(constraint(at), FixedCell)
   P = DirectPrecon(p, speye(2), copy(positions(at)),
                     updatedist, tol, updatefreq, 0, stab)
   return force_update!(P, at)
end


A_ldiv_B!(out::Dofs, P::AMGPrecon, x::Dofs) = A_ldiv_B!(out, P.amg, x)
A_mul_B!(out::Dofs, P::AMGPrecon, f::Dofs) = A_mul_B!(out, P.amg, f)

A_ldiv_B!(out::Dofs, P::DirectPrecon, x::Dofs) = A_ldiv_B!(out, P.A, x)
A_mul_B!(out::Dofs, P::DirectPrecon, f::Dofs) = A_mul_B!(out, P.A, f)

dot(x, P::Union{DirectPrecon, AMGPrecon}, y) = dot(x, P * y)

*(P::DirectPrecon, x::AbstractVector) = P.A * x
*(P::AMGPrecon, x::AbstractVector) = P.amg * x
\(P::DirectPrecon, x::AbstractVector) = P.A \ x
\(P::AMGPrecon, x::AbstractVector) = P.amg \ x


need_update(P::PairPrecon, at::AbstractAtoms) =
   (P.skippedupdates > P.updatefreq) ||
   (maxdist(positions(at), P.oldX) >= P.updatedist)

update!(P::PairPrecon, at::AbstractAtoms) =
   need_update(P, at) ? force_update!(P, at) : (P.skippedupdates += 1; P)


function force_update!(P::AMGPrecon, at::AbstractAtoms)
   # perform updates of the potential p (if needed; usually not)
   P.p = update_inner!(P.p, at)
   # construct the preconditioner matrix ...
   Pmat = precon_matrix(P.p, at)
   Pmat + P.stab * speye(size(Pmat, 1))
   A = project!( constraint(at), Pmat )
   # and the AMG solver
   P.amg = RugeStubenSolver(A, tol=P.tol)
   # remember the atom positions
   copy!(P.oldX, positions(at))
   # and remember that we just did a full update
   P.skippedupdates = 0
   return P
end

function force_update!(P::DirectPrecon, at::AbstractAtoms)
   # perform updates of the potential p (if needed; usually not)
    P.p = update_inner!(P.p, at)
   # construct the preconditioner matrix ...
   Pmat = precon_matrix(P.p, at)
   Pmat + P.stab * speye(size(Pmat, 1))
   P.A = project!( constraint(at), Pmat )
   # remember the atom positions
   copy!(P.oldX, positions(at))
   # and remember that we just did a full update
   P.skippedupdates = 0
   return P
end

# ============== some tools to construct preconditioners ======================

function estimate_rnn(at::AbstractAtoms)
   sym = unique(chemical_symbols(at))
   return minimum([rnn(s) for s in sym])
end


# ============== Implementation of some concrete types ======================

# default implementation of update_inner!;
# most of the time no inner update is required

"update `AMGPrecon.p`, must return updated `p`"
update_inner!(p::Any, at::AbstractAtoms) = p

"return the three linear indices associated with an atom index"
atind2lininds(i::Integer) = (i-1) * 3 + [1;2;3]

"""
build the preconditioner matrix associated with a pair potential;
this is related to but not even close to the same as the hessian matrix!
"""
function precon_matrix(p::PairPotential, at::AbstractAtoms)
   I = Int[]; J = Int[]; Z = Float64[]
   for (i, j, r, _) in pairs(at, cutoff(p))
      # the next 2 lines add an identity block for each atom
      # TODO: should experiment with other matrices, e.g., R ⊗ R
      ii = atind2lininds(i)
      jj = atind2lininds(j)
      z = p(r)
      for (a, b) in zip(ii, jj)
         append!(I, [a; a; b; b])
         append!(J, [a; b; a; b])
         append!(Z, [z; -z; -z; z])
      end
   end
   N = 3*length(at)
   return sparse(I, J, Z, N, N) + 0.001 * speye(N)
end


"""
A variant of the `Exp` preconditioner; see

### Constructor: `Exp(at::AbstractAtoms)`

Keyword arguments:

* `A=3.0`: stiffness of potential
* `r0=nothing`: if `nothing`, then it is an estimate nn distance
* `cutoff_mult`: cut-off multiplier
* `tol, updatefrew`: AMG parameters

### Reference

      D. Packwood, J. Kermode, L. Mones, N. Bernstein, J. Woolley, N. I. M. Gould,
      C. Ortner, and G. Csanyi. A universal preconditioner for simulating condensed
      phase materials. J. Chem. Phys., 144, 2016.
"""
function Exp(at::AbstractAtoms;
             A=3.0, r0=estimate_rnn(at), cutoff_mult=2.2,
             tol=1e-7, updatefreq=10, solver = :amg, energyscale = 1.0)
   e0 = energyscale == :auto ? 1.0 : energyscale
   rcut = r0 * cutoff_mult
   exp_shift = e0 * exp( - A*(rcut/r0 - 1.0) )
   pot = let e0=e0, A=A, r0=r0, exp_shift = e0 * exp( - A*(rcut/r0 - 1.0) )
      (@analytic( r -> e0 * exp( - A * (r/r0 - 1.0)) - exp_shift)) * HS(rcut)
   end
   if solver == :amg
      P = AMGPrecon(pot, at, updatedist=0.2 * r0, tol=tol, updatefreq=updatefreq)
   elseif solver == :direct
      P = DirectPrecon(pot, at, updatedist=0.2 * r0, tol=tol, updatefreq=updatefreq)
   else
      error("unknown kwarg solver = $(solver)")
   end

   if energyscale == :auto
      e0 = estimate_energyscale(at, P)
      P = Exp(at, A=A, r0=r0, cutoff_mult=cutoff_mult, tol=tol,
               updatefreq=updatefreq, solver=solver, energyscale=e0)
   end

   return P
end

# want μ * <P v, v> ~ <∇E(x+hv) - ∇E(x), v> / h
function estimate_energyscale(at, P)
   # get the P-matrix at current configuration
   A = precon_matrix(P.p, at)
   # determine direction in which the cell is maximal
   F = Matrix(defm(at))
   X0 = positions(at)
   _, i = findmax( norm(F[:, j]) for j = 1:3 )
   Fi = JVecF(F[:, i])
   # an associated perturbation
   V = [sin(2*pi * dot(Fi, x)/dot(Fi,Fi)) * Fi for x in X0]
   # compute gradient at current positions
   f0 = forces(at) |> mat
   # compute gradient at perturbed positions
   h = 1e-3
   set_positions!(at, X0 + h * V)
   f1 = forces(at) |> mat
   # return the estimated value for the energyscale
   V = mat(V)
   μ = - vecdot((f1 - f0)/h, V) / dot(A * V[:], V[:])
   if μ < 0.1 || μ > 100.0
      warn("""
      e0 = $(μ) in `estimate_energyscale`; this is likely due to a poor \
      starting guess in an optimisation, probably best to set the
      energyscale manually
      """)
   end
   return μ
end


include("ffprecond.jl")

end # end module Preconditioners
