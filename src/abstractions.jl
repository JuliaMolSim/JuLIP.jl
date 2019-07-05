
import LinearAlgebra: ldiv!, mul!
using LinearAlgebra: det, UniformScaling
import Base: length, getindex, setindex!, deleteat!

# export AbstractAtoms, AbstractCalculator, AbstractConstraint, NullConstraint,
#        Preconditioner, NullCalculator

# function defined primarily on AbstractAtoms
export positions, get_positions, set_positions!,
       momenta, get_momenta, set_momenta!,
       masses, get_masses, set_masses,
       cell, get_cell, set_cell!, is_cubic,
       pbc, get_pbc, set_pbc!,
       set_data!, get_data, has_data,
       set_calculator!, calculator, get_calculator,
       set_constraint!, constraint, get_constraint,
       neighbourlist, cutoff,
       apply_defm!,
       energy, potential_energy, forces, gradient, hessian,
       site_energies, site_energy, partial_energy,
       stress, virial, site_virials,
       position_dofs, set_position_dofs!,
       momentum_dofs, set_momentum_dofs!,
       dofs, set_dofs!,
       preconditioner,
       volume

# temporary prototypes while rewriting 
export rattle!
function rattle! end


# here we define and document the prototypes that are implemented

"""
`AbstractAtoms{T}`

the abstract supertype for storing atomistic configurations. A basic
implementation might simply store a list of positions, see e.g. `Atoms`.

Some Base functions that should be overloaded for atoms objects:
- length
- deleteat
- getindex
- setindex!
"""
abstract type AbstractAtoms{T} end

abstract type AbstractConstraint end

"""
`AbstractCalculator`: theabstractsupertype of calculators. These
store model information, and are linked to the implementation of energy,
forces, and so forth.
"""
abstract type AbstractCalculator end



"""
`Dofs{T}`: dof vector; simply an alias for `Vector{T}`
"""
const Dofs{T} = Vector{T}


# TODO: probably rename Preconditioner to AbstractPreconditioner and
#       AMGPrecon to Preconditioner


"""
`positions(at)` : Return copy of positions of all atoms as a `Vector{JVec}`
"""
function positions end

"alias for `positions`"
get_positions = positions

"`set_positions!(at, X) -> at` : Set positions of all atoms"
function set_positions! end

set_positions!(at::AbstractAtoms, p::AbstractMatrix) = set_positions!(at, vecs(p))
set_positions!(at::AbstractAtoms, x, y, z) = set_positions!(at, [x'; y'; z'])

# this is already documented at its first definition in `arrayconversions`?
xyz(at::AbstractAtoms) = xyz(positions(at))


"`momenta(at)` : Return copy of momenta of all atoms as a `Vector{<:JVec}`."
function momenta end

"alias for `momenta`"
get_momenta = momenta

"`set_momenta!(at, P) -> at` : Set momenta of all atoms"
function set_momenta! end

set_momenta!(at::AbstractAtoms, p::AbstractMatrix) = set_momenta!(at, vecs(p))

"`masses` : return vector of all masses"
function masses end

"`get_masses` : alias for `masses`"
get_masses = masses

"`set_masses!(at, M) -> at` : set the atom masses"
function set_masses! end

"`cell(at) -> JMat` : get computational cell (the rows are the lattice vectors)"
function cell end

"alias for `cell`"
get_cell = cell

"`set_cell!(at, C) -> at` : set computational cell; cf. `?cell`"
function set_cell! end

"`volume(at)` : return volume of computational cell"
volume(at) = abs(det(cell(at)))

"""
`apply_defm!(at, F, t = zero(JVec)) -> at` : for a 3 x 3 matrix `F` and
`at::AbstractAtoms` the affine deformation `x -> Fx + t` is applied
to the configuration, modifying `at` inplace and returning it. This
modifies both the cell and the positions.
"""
function apply_defm!(at::AbstractAtoms, F::AbstractMatrix, t::AbstractVector)
   @assert size(F) == (3,3)
   @assert length(T) == 3
   C = cell(at)
   X = positions(at)
   Cnew = C * F'
   for n = 1:length(X)
      # TODO: project back into the cell?
      X[n] = F * X[n] + t
   end
   set_positions!(at, X)
   set_cell!(at, C)
   return at
end


"`set_pbc!(at, p) -> at` : set periodic boundary conditions"
function set_pbc! end
set_pbc!(at::AbstractAtoms, p::Bool) = set_pbc!(at, (p,p,p))

"`pbc(at)` : get array or tuple determining which directions are periodic"
function pbc end

"`get_pbc(at)` : alias for `pbc`"
get_pbc = pbc

"""
`is_cubic(at) -> Bool` : determines whether a cubic cell is used
(i.e. cell is a diagonal matrix)
"""
is_cubic(a::AbstractAtoms) = isdiag(cell(a))


"""
`set_data!(at, name, value)`:
associate some data with `at`; to be stored in a Dict within `at`

if `name::Union{Symbol, AbstractString}`, then `setindex!` is an alias
for `set_data!`, i.e., we may write `at[:id] = val`
"""
function set_data! end
setindex!(at::AbstractAtoms, value,
          name::Union{Symbol, AbstractString}) = set_data!(at, name, value)

"""
`get_data(at, name)`:
obtain some data stored with `set_data!`

if `name::Union{Symbol, AbstractString}`, then `getindex` is an alias
for `get_data`.
"""
function get_data end

getindex(at::AbstractAtoms,
         name::Union{Symbol, AbstractString}) = get_data(at, name)

"`has_data(at, key)` : check whether some data with id `key` is already stored"
function has_data end

"alias for `deleteat!`"
delete_atom! = deleteat!

"`calculator(at)` : return an attached calculator"
function calculator end

"`set_calculator!(at, calc) -> at` : attach a calculator"
function set_calculator! end

"`constraint(at)` : return attached constraint"
function constraint end

"`set_constraint!(at, cons) -> at` : attach a constraint"
function set_constraint! end


"""
`neighbourlist(at, rcut)`

construct a suitable neighbourlist for this atoms object. `rcut` should
be allowed to be either a scalar or a vector.
"""
function neighbourlist end



#######################################################################
#     CALCULATOR
#######################################################################

struct NullCalculator <: AbstractCalculator end

"""
`cutoff(calc)` : Returns the cut-off radius of the attached potential.

`cutoff(at)` : returns the cutoff of the attached calculator
"""
function cutoff end

cutoff(at::AbstractAtoms) = cutoff(calculator(at))

"""
`energy(calc, at)`: Return the total potential energy of a configuration of
atoms `at`, using the calculator `calc`. In addition, `energy` can be
called in the following ways:
*  `energy(calc, at)`: base definition; this is normally the only method that
     needs to be overloaded when a new calculator is implemented.
* `energy(at)`: if a calculator is attached to `at`
* `energy(calc, at, const, dof)`
* `energy(calc, at, dof)`
* `energy(at, dofs)`
"""
function energy end
energy(at::AbstractAtoms) = energy(calculator(at), at)
potential_energy = energy

# TODO : move the aliases for energy to here

"""
`site_energy(calc, at, n)` : return site energy at atom idx `n`
"""
function site_energy end
site_energy(at::AbstractAtoms, n::Integer) = site_energy(calculator(at), at, n)

"""
`site_energies(calc, at)` : return vector of all site energies
"""
function site_energies end
site_energies(a::AbstractAtoms) = site_energies(calculator(a), a)

"""
`partial_energy(calc, at, subset)` : return energy contained in a `subset` of the
configuration
"""
function partial_energy end
partial_energy(at::AbstractAtoms, subset) =
      partial_energy(calculator(at), at, subset)

"""
`forces(calc, at)` : forces as `Vector{<:JVec}`  (negative gradient w.r.t. atom positions only)
"""
function forces end
forces(at::AbstractAtoms) = forces(calculator(at), at)

"""
`hessian_pos(calc, at):` block-hessian with respect to all atom positions
"""
function hessian_pos end
hessian_pos(at::AbstractAtoms) = hessian_pos(calculator(at), at)

"""
* `virial(c::AbstractCalculator, a::AbstractAtoms) -> JMat`
* `virial(a::AbstractAtoms) -> JMat`

returns virial, (- ∂E / ∂F) where `F = cell(a)'`
"""
function virial end
virial(a::AbstractAtoms) = virial(calculator(a), a)

"""
* `stress(c::AbstractCalculator, a::AbstractAtoms) -> JMatF`
* `stress(a::AbstractAtoms) -> JMatF`

stress = - virial / volume; this function should *not* be overloaded;
instead overload virial
"""
stress(calc::AbstractCalculator, at::AbstractAtoms) =
      - virial(calc, at) / volume(a)

stress(at::AbstractAtoms) = stress(calculator(at), at)


#######################################################################
#  Constraints and DoF Handling
#######################################################################

struct NullConstraint <: AbstractConstraint end

"""
`position_dofs(at::AbstractAtoms, cons::AbstractConstraint) -> Dofs`
`position_dofs(at::AbstractAtoms) -> Dofs`
`dofs(at::AbstractAtoms, cons::AbstractConstraint) -> Dofs`

Take an atoms object `at` and return a Dof-vector that fully describes the
state given the constraint `cons`
"""
function position_dofs end
position_dofs(at::AbstractAtoms) = position_dofs(at, constraint(at))

"""
`dofs` is an alias for `position_dofs`
"""
dofs(args...) = position_dofs(args...)

"""
* `set_position_dofs!(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs) -> at`
* `set_position_dofs!(at::AbstractAtoms, x::Dofs) -> at`
* `set_dofs!(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs) -> at`
* `set_dofs!(at::AbstractAtoms, cons::AbstractConstraint) -> at`

change configuration stored in `at` according to `cons` and `x`.
"""
function set_position_dofs! end
set_position_dofs!(at::AbstractAtoms, x::Dofs) =
      set_position_dofs!(at, constraint(at), x)

"""
`set_dofs!` is an alias for `set_position_dofs!`
"""
set_dofs!(args...) = set_position_dofs!(args...)


"""
`momentum_dofs(at::AbstractAtoms, cons::AbstractConstraint) -> Dofs`
`momentum_dofs(at::AbstractAtoms) -> Dofs`

Take an atoms object `at` and return a Dof-vector that fully describes the
momenta (given the constraint `cons`)
"""
function momentum_dofs end
momentum_dofs(at::AbstractAtoms) = momentum_dofs(at, constraint(at))

"""
* `set_momentum_dofs!(at::AbstractAtoms, cons::AbstractConstraint, p::Dofs) -> at`
* `set_momentum_dofs!(at::AbstractAtoms, p::Dofs) -> at`

change configuration stored in `at` according to `cons` and `p`.
"""
function set_momentum_dofs! end
set_momentum_dofs!(at::AbstractAtoms, q::Dofs) =
      set_momentum_dofs!(at, constraint(at), q)

"""
* `project!(at::AbstractAtoms, cons::AbstractConstraint) -> at`
* `project!(at::AbstractAtoms) -> at`

project the `at` onto the constraint manifold.
"""
function project! end

project!(at::AbstractAtoms) = project!(at, constraint(at))


# converting calculator functionality

"""
* `energy(at, x::Dofs) -> Float64`

`potential_energy`; with potentially added cell terms, e.g., if there is
applied stress or applied pressure.
"""
energy(at::AbstractAtoms, x::Dofs) = energy(set_dofs!(at, x))
energy(calc::AbstractCalculator, at::AbstractAtoms, x::Dofs) = energy(calc, set_dofs!(at, x))

"""
* `gradient(calc, at, cons::AbstractConstraint, c::Dofs) -> Dofs`
* `gradient(at) -> Dofs`
* `gradient(at, x::Dofs) -> Dofs`

gradient of `potential_energy` in dof-format;
depending on the constraint this could include e.g. a gradient w.r.t. cell
shape; normally only the form `gradient(calc, at, cons)` should be
overloaded by an `AbstractConstraints` object.
"""
function gradient end
gradient(at::AbstractAtoms, x::Dofs) =
      gradient(set_dofs!(at, x))
gradient(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs) =
      gradient(set_dofs!(at, cons, x), cons)
gradient(calc::AbstractCalculator, at::AbstractAtoms, x::Dofs) =
      gradient(calc, set_dofs!(at, x))
gradient(at::AbstractAtoms) =
      gradient(calculator(at), at, constraint(at))
gradient(calc::AbstractCalculator, at::AbstractAtoms) =
      gradient(calc, at, constraint(at))
gradient(at::AbstractAtoms, cons::AbstractConstraint) =
      gradient(calculator(at), at, cons)


"""
`hessian`: compute hessian of total energy with respect to DOFs!

same as gradient, just returns the hessian; new constraints should only
overload `hessian(calc, at, cons)`
"""
function hessian end
hessian(at::AbstractAtoms, x::Dofs) =
      hessian(set_dofs!(at, x))
hessian(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs) =
      hessian(set_dofs!(at, cons, x), cons)
hessian(calc::AbstractCalculator, at::AbstractAtoms, x::Dofs) =
      hessian(calc, set_dofs!(at, x))
hessian(at::AbstractAtoms) =
      hessian(calculator(at), at, constraint(at))
hessian(calc::AbstractCalculator, at::AbstractAtoms) =
      hessian(calc, at, constraint(at))
hessian(at::AbstractAtoms, cons::AbstractConstraint) =
      hessian(calculator(at), at, cons)


#######################################################################
#                     PRECONDITIONER
# we don't create an abstract type, but somewhere we should
# document that a preconditioner must implement the following
# operations:
#   - update!(precond, at)
#   - ldiv!(out, P, f)
#   - mul!)(out, P, x)
#######################################################################

"""
`update!(precond, at)`

Update the preconditioner with the new geometry information.
"""
function update! end
update!(precond, at::AbstractAtoms, x::Dofs) =
            update!(precond, set_dofs!(at, x))

# TODO: make a Julia PR? do this in-place
ldiv!(out::Dofs, P::UniformScaling, x::Dofs) =
      copyto!(out, P \ x)

# mul! is already correctly defined -> no need to define it here.
update!(P::UniformScaling, at::AbstractAtoms) = P

"""
construct a preconditioner suitable for this atoms object

default is `I`
"""
preconditioner(at::AbstractAtoms) =
            preconditioner(at, calculator(at), constraint(at))

preconditioner(at::AbstractAtoms, calc::AbstractCalculator,
               con::AbstractConstraint) = I
