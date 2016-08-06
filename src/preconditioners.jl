
module Preconditioners

import JuLIP: Preconditioner, JPts, Dofs, update!, maxdist
import PyAMG: RugeStubenSolver
import Base: A_ldiv_B!, A_mul_B!
import JuLIP.Potentials: PairPotential, AnalyticPotential
import JuLIP.Constraints: project!, FixedCell
import JuLIP.ASE: chemical_symbols, rnn

# ================ AMGPrecon =====================
# this is wrapping some machinery around PyAMG

"""
`AMGPrecon{T}`: a preconditioner using AMG as the main solver

`AMGPrecon` stores a field `p` which is used to determine the preconditioner
matrix via
```julia
matrix(P, at::AbstractAtoms)
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
type AMGPrecon{T}
   p::T
   amg::RugeStubenSolver
   oldX::JPts
   updatedist::Float64
   tol::Float64
   updatefreq::Int
   skippedupdates::Int
end


function AMGPrecon(p, at; updatedist=0.3, tol=1e-7, updatefreq=10)
   # make sure we don't use this in a context it is not intended for!
   @assert isa(constraint(at), FixedCell)
   return force_update!(AMGPrecon(p, RugeStubenSolver(sparse(eye(2))), JPts(0),
                                 updatedist=updatedist, tol=tol, updatefreq, 0),
                        at)
end


A_ldiv_B!(out::Dofs, P::AMGPrecon, x::Dofs) = A_ldiv_B!(out, P.amg, x)
A_mul_B!(out::Dofs, P::AMGPrecon, f::Dofs) = A_mul_B!(out, P.amg, f)

need_update(P::AMGPrecon, at::AbstractAtoms) =
   (P.skippedupdates > P.updatefreq) ||
   (maxdist(ositions(at), P.oldX) >= P.updatedist)

update!(P::AMGPrecon, at::AbstractAtoms) =
   need_update(P, at) ? force_update!(P, at) : (P.skippedupdates += 1; P)


function force_update!(P::AMGPrecon, at::AbstractAtoms)
   # perform updates of the potential p (if needed; usually not)
   P.p = update_inner!(P.p, at)
   # construct the preconditioner matrix ...
   A = project!( at.constraint(), matrix(P.p, at) )
   # and the AMG solver
   P.amg = RugeStubenSolver(A, tol=P.tol)
   # remember the atom positions
   copy!(P.oldX, positions(at)))
   # and remember that we just did a full update
   P.skippedupdates = 0
   return P
end

# ============== some tools to construct preconditioners ======================

function estimate_rnn(at:AbstractAtoms)
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
function matrix(p::PairPotential, at::AbstractAtoms)
   I = Int[]; J = Int[]; Z = Float64[]
   for (i, j, r, _, _) in bonds(at, cutoff(p))
      # the next 2 lines add an identity block for each atom
      # TODO: should experiment with other matrices, e.g., R ⊗ R
      ii = atind2lininds(i); jj = atind2lininds(j)
      append!(I, ii); append!(J, jj); append!(Z, ones(3) * p(r))
   end
   return sparse(I, J, Z)
end


"""
A variant of the `Exp` preconditioner; see

### Constructor: `Exp(at::AbstractAtoms)`

Keyword arguments:
* `A=3.0`: stiffness of potential
* `rnn=nothing`: `rnn` is then estimated
* `cutoff_mult`: cut-off multiplier
* `tol, updatefrew`: AMG parameters

### Reference

D. Packwood, J. Kermode, L. Mones, N. Bernstein, J. Woolley, N. I. M. Gould, C. Ortner, and G. Csanyi. A universal preconditioner for simulating condensed phase materials. J. Chem. Phys., 144, 2016.
"""
function Exp(at::AbstractAtoms;
            A=3.0, rnn=nothing, cutoff_mult=2.2, tol=1e-7, updatefreq=10)
   if rnn == nothing
      rnn = estimate_rnn(at)
   end
   cutoff = rnn * cutoff_mult
   exp_shit = exp( - A*(cutoff/rnn - 1.0) )
   pot = AnalyticPotential( :(exp( - $A * (r/$rnn - 1.0)) - $exp_shit),
                            cutoff = cutoff )
   return AMGPrecon(pot, at, updatedist=0.3 * rnn,
                  tol=tol, updatefreq=updatefreq)
end
