
module Preconditioners

using JuLIP: AbstractAtoms, Preconditioner, JVecs, JVecsF, Dofs, maxdist,
            constraint, pairs, cutoff, positions, defm, JVecF, forces, mat, vecs,
            set_positions!, julipwarn, chemical_symbols, rnn, JVec, JMat ,
            AbstractCalculator, calculator

using JuLIP.Potentials: @analytic, evaluate, evaluate_d, PairPotential, HS,
         SitePotential, sites, C0Shift, _precon_or_hessian_pos

using JuLIP.Constraints: project!, FixedCell, _pos_to_alldof

using AMG


import JuLIP: update!
import JuLIP.Potentials: precon, cutoff
import Base: A_ldiv_B!, A_mul_B!, dot, *, \, size

export Exp, FF



"""
`IPPrecon`: the standard preconditioner type for JuLIP

`IPPrecon` stores a field `p <: AbstractCalculator` which is
should implement `precon(p, r, R)` which is a block matrix analogue of
`hess(p, r, R)` and is used to assemble the preconditioner matrix via
```julia
precon_matrix(P, at::AbstractAtoms)
```
If `p` should be updated in response to an update of `at` then
the type can overload `JuLIP.Preconditioners.update_inner!`.

## Constructor:
```
IPPrecon(p::AbstractCalculator, at::AbstractAtoms; kwargs...)
```

## Keyword arguments:

* `updatedist`: determines after how much movement of the atoms, the
preconditioner is updated
* `tol`: solver tolerance if AMG is used
* `updatefreq`: after how many iterations do we force an update {10}
* `stab`: stabilisation constant {0.01}, add `stab * I` to `P`
* `solve`: which solver to used, `:amg` or `:chol`, default is `:amg`
"""
mutable struct IPPrecon{TV, TS, T <: AbstractFloat, TI <: Integer} <: Preconditioner
   p::TV
   solver::TS
   A::SparseMatrixCSC{T, TI}
   oldX::Vector{JVec{T}}
   updatedist::T
   tol::T
   updatefreq::TI
   skippedupdates::TI
   stab::T
   innerstab::T
end



function IPPrecon(p::AbstractCalculator, at::AbstractAtoms;
         updatedist=0.2 * rnn(at), tol=1e-7, updatefreq=10, stab=0.01,
         solver = :chol, innerstab=0.0)
   # make sure we don't use this in a context it is not intended for!
   @assert isa(constraint(at), FixedCell)
   A = AMG.poisson(12)
   if solver == :amg
      solver = ruge_stuben(A)
   elseif solver in [:chol, :direct]
      solver = cholfact(A)
   else
      error("`IPPrecon` : unknown solver $(solver)")
   end
   P = IPPrecon(p, solver, A, positions(at), updatedist, tol, updatefreq, 0, stab, innerstab)
   # the force_update! makes the first assembly pass.
   return force_update!(P, at)
end

# some standard Base functionality lifted to IPPrecon
A_ldiv_B!(out::Dofs, P::IPPrecon{TV, TS}, f::Dofs) where TV where TS <: AMG.MultiLevel =
   copy!(out, solve(P.solver, f; maxiter = 200, tol = P.tol))
A_ldiv_B!(out::Dofs, P::IPPrecon{TV, TS}, f::Dofs) where TV where TS <: Base.SparseArrays.CHOLMOD.Factor =
   copy!(out, P.solver \ f)
A_mul_B!(out::Dofs, P::IPPrecon, x::Dofs) = A_mul_B!(out, P.A, x)
dot(x, P::IPPrecon, y) = dot(x, P * y)
*(P::IPPrecon, x::AbstractVector) = P.A * x
\(P::IPPrecon, x::AbstractVector) = A_ldiv_B!(zeros(length(x)), P, x)
Base.size(P::IPPrecon) = size(P.A)


need_update(P::IPPrecon, at::AbstractAtoms) =
   (P.skippedupdates > P.updatefreq) ||
   (maxdist(positions(at), P.oldX) >= P.updatedist)

update!(P::IPPrecon, at::AbstractAtoms) =
   need_update(P, at) ? force_update!(P, at) : (P.skippedupdates += 1; P)


update_solver!(P::IPPrecon{TV, TS}) where TV where TS <: Base.SparseArrays.CHOLMOD.Factor =
   cholfact(Symmetric(P.A))

update_solver!(P::IPPrecon{TV, TS}) where TV where TS <: AMG.MultiLevel =
   ruge_stuben(P.A)


function force_update!(P::IPPrecon, at::AbstractAtoms)
   # perform updates of the potential p (if needed; usually not)
   P.p = update_inner!(P.p, at)
   # construct the preconditioner matrix ...
   Pmat = precon_matrix(P.p, at, P.innerstab)
   Pmat = Pmat + P.stab * speye(size(Pmat, 1))
   Pmat = project!(constraint(at), Pmat)
   # and the AMG solver
   P.A = Pmat
   P.solver = update_solver!(P)
   # remember the atom positions
   copy!(P.oldX, positions(at))
   # and remember that we just did a full update
   P.skippedupdates = 0
   return P
end




# ============== Implementation of some concrete types ======================

# default implementation of update_inner!;
# most of the time no inner update is required

"update `IPPrecon.p`, must return updated `p`"
update_inner!(p::Any, at::AbstractAtoms) = p

"return the three linear indices associated with an atom index"
atind2lininds(i::Integer) = (i-1) * 3 + [1;2;3]

"""
build the preconditioner matrix associated with the potential V
"""
precon_matrix(V::AbstractCalculator, at::AbstractAtoms, innerstab=0.0;
              preconmap = (V, r, R) -> precon(V, r, R, innerstab)) =
   _pos_to_alldof(_precon_or_hessian_pos(V, at, preconmap), at)

"""
A variant of the `Exp` preconditioner; see

### Constructor: `Exp(at::AbstractAtoms)`

Keyword arguments:

* `A`: stiffness of potential {default 3.0}
* `r0`: nn distance estimate {default is an automicatic estimate}
* `cutoff_mult`: cut-off is `r0 * cutoff_mult` {default 2.2}
* `kwargs...`: `IPPrecon` parameters

### Reference

      D. Packwood, J. Kermode, L. Mones, N. Bernstein, J. Woolley, N. I. M. Gould,
      C. Ortner, and G. Csanyi. A universal preconditioner for simulating condensed
      phase materials. J. Chem. Phys., 144, 2016.
"""
mutable struct Exp{T, TV} <: PairPotential
   Vexp::TV
   energyscale::T
end

cutoff(P::Exp) = cutoff(P.Vexp)

precon{T}(P::Exp{T}, r, R, innerstab) = (P.energyscale * P.Vexp(r)) * one(JMat{T})

function Exp(at::AbstractAtoms;
             A=3.0, r0=rnn(at), cutoff_mult=2.2, energyscale = 1.0,
             kwargs...)
   e0 = energyscale == :auto ? 1.0 : energyscale
   rcut = r0 * cutoff_mult
   Vexp = let A=A, r0=r0
      @analytic r -> exp( - A * (r/r0 - 1.0))
   end * C0Shift(rcut)
   P = IPPrecon(Exp(Vexp, e0), at; kwargs...)
   if energyscale == :auto
      P.p.energyscale = estimate_energyscale(at, P)
      force_update!(P, at)
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
      e0 = $(μ) in `estimate_energyscale`; this is likely due to a poor
      starting guess in an optimisation, probably best to set the
      energyscale manually, or use CG instead of LBFGS.
      """)
   end
   return μ
end





"""
`FF`: defines a preconditioner based on a force-field;

TODO: thorough documentation and reference once the paper is finished
"""
FF(at::AbstractAtoms, V::AbstractCalculator; kwargs...) =
      IPPrecon(V, at; kwargs...)

FF(at::AbstractAtoms; kwargs...) =
      FF(at, calculator(at); kwargs...)



end # end module Preconditioners
