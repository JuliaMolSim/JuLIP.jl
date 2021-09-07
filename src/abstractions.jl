
import LinearAlgebra: ldiv!, mul!
using LinearAlgebra: det, UniformScaling, isdiag
import Base: length, getindex, setindex!, deleteat!

import NeighbourLists: cutoff

import ACEbase: fltype, fltype_intersect, rfltype,
                evaluate, evaluate_d, evaluate_dd, evaluate_ed,
                evaluate!, evaluate_d!, evaluate_dd!, evaluate_ed!,
                precon!


# function defined primarily on AbstractAtoms
export positions, get_positions, set_positions!,
       momenta, get_momenta, set_momenta!,
       masses, get_masses, set_masses,
       cell, get_cell, set_cell!, is_cubic, cell_vecs,
       pbc, get_pbc, set_pbc!,
       set_data!, get_data, has_data,
       set_calculator!, calculator, get_calculator,
       neighbourlist, cutoff,
       apply_defm!,
       energy, potential_energy, forces, gradient, hessian,
       dipole, 
       site_energies,
       site_energy, partial_energy,
       site_energy_d, partial_energy_d,
       stress, virial, site_virials,
       position_dofs, set_position_dofs!,
       momentum_dofs, set_momentum_dofs!,
       dofs, set_dofs!,
       preconditioner,
       volume,
       chemical_symbols,
       atomic_numbers,
       AbstractAtoms, AbstractCalculator


# temporary prototypes while rewriting


# todo: prototype and document these here
function chemical_symbols end
function atomic_numbers end
function site_energy_d end
function partial_energy_d end


"""
`function dipole end ` : prototype for a global dipole function, 
not currently implemented in JuLIP.
"""
function dipole end 



"""
`alloc_temp(args...)` : allocate temporary arrays for the evaluation of
some calculator or potential; see developer docs for more information
"""
alloc_temp(args...) = nothing

"""
`alloc_temp_d(args...)` : allocate temporary arrays for the evaluation of
some calculator or potential; see developer docs for more information
"""
alloc_temp_d(args...) = nothing

alloc_temp_dd(args...) = nothing




# -----


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

fltype(at::AbstractAtoms{T}) where {T} = T

abstract type AbstractConstraint end

"""
`AbstractCalculator`: theabstractsupertype of calculators. These
store model information, and are linked to the implementation of energy,
forces, and so forth.
"""
abstract type AbstractCalculator end

fltype(V::AbstractCalculator) = Float64  # a sensible default!

"""
`Dofs{T}`: dof vector; simply an alias for `Vector{T}`
"""
const Dofs{T} = Vector{T}


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

"""
`cell_vecs(at) -> (SVec, SVec, SVec)` : return the three cell vectors
"""
function cell_vecs(at::AbstractAtoms)
   C = cell(at)
   @assert size(C) == (3,3)
   return C[1,:], C[2,:], C[3,:]
end


"`volume(at)` : return volume of computational cell"
volume(at) = abs(det(cell(at)))

"""
`apply_defm!(at, F, t = zero(JVec)) -> at` : for a 3 x 3 matrix `F` and
`at::AbstractAtoms` the affine deformation `x -> Fx + t` is applied
to the configuration, modifying `at` inplace and returning it. This
modifies both the cell and the positions.
"""
function apply_defm!(at::AbstractAtoms{T}, F::AbstractMatrix,
                     t::AbstractVector = zero(JVec{T})) where {T}
   @assert size(F) == (3,3)
   @assert length(t) == 3
   C = cell(at)
   X = positions(at)
   Cnew = C * F'
   for n = 1:length(X)
      # TODO: project back into the cell?
      X[n] = F * X[n] + t
   end
   set_cell!(at, Cnew)
   set_positions!(at, X)
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


"""
`neighbourlist(at, rcut)`

construct a suitable neighbourlist for this atoms object. `rcut` should
be allowed to be either a scalar or a vector.
"""
function neighbourlist end


"""
`static_neighbourlist(at::AbstractAtoms, cutoff; key = :staticnlist)`

This function first checks whether a static neighbourlist already exists
with cutoff `cutoff` and if it does then it returns the existing list.
If it does not, then it computes a new neighbour list with the current
configuration, stores it for later use and returns it.
"""
function static_neighbourlist(at::AbstractAtoms, cutoff; key=:staticnlist)
   recompute = false
   if has_data(at, key)
      nlist = get_data(at, key)
      if cutoff(nlist) != cutoff
         recompute = true
      end
   else
      recompute = true
   end
   if recompute
      set_data!( at, key, neighbourlist(at, cutoff) )
   end
   return get_data(at, key)
end

static_neighbourlist(at::AbstractAtoms) = static_neighbourlist(at, cutoff(at))


#######################################################################
#     CALCULATOR
#######################################################################

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
potential_energy = energy

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
virial(at::AbstractAtoms) = virial(calculator(at), at)

"""
* `stress(c::AbstractCalculator, a::AbstractAtoms) -> JMatF`
* `stress(a::AbstractAtoms) -> JMatF`

stress = - virial / volume; this function should *not* be overloaded;
instead overload virial
"""
stress(calc::AbstractCalculator, at::AbstractAtoms) =
      - virial(calc, at) / volume(at)

stress(at::AbstractAtoms) = stress(calculator(at), at)


#######################################################################
#  Constraints and DoF Handling
#######################################################################


"""
`position_dofs(at::AbstractAtoms) -> Dofs`
`dofs(at::AbstractAtoms) -> Dofs`

Take an atoms object `at` and return a Dof-vector that fully describes the
state given the potential constraints placed on it.
"""
function position_dofs end

"""
`dofs` is an alias for `position_dofs`
"""
dofs = position_dofs

"""
* `set_position_dofs!(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs) -> at`
* `set_position_dofs!(at::AbstractAtoms, x::Dofs) -> at`
* `set_dofs!(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs) -> at`
* `set_dofs!(at::AbstractAtoms, cons::AbstractConstraint) -> at`

change configuration stored in `at` according to `cons` and `x`.
"""
function set_position_dofs! end

"""
`set_dofs!` is an alias for `set_position_dofs!`
"""
set_dofs! = set_position_dofs!


"""
`momentum_dofs(at::AbstractAtoms) -> Dofs`

Take an atoms object `at` and return a Dof-vector that fully describes the
momenta (given any constraints placed on `at`)
"""
function momentum_dofs end

"""
* `set_momentum_dofs!(at::AbstractAtoms, cons::AbstractConstraint, p::Dofs) -> at`
* `set_momentum_dofs!(at::AbstractAtoms, p::Dofs) -> at`

change configuration stored in `at` according to `cons` and `p`.
"""
function set_momentum_dofs! end

"""
* `project!(at::AbstractAtoms) -> at`

project the `at` onto the constraint manifold.
"""
function project! end


# converting calculator functionality
# The following lines define variants of the `energy` function, which
# get around the issue that energy may or may not include an external
# potential; the rule is as follows:
#  - a new calculator must implement `energy(calc, at)`
#  - a new constraint must implement `energy(calc, at, cons)`
#    and at the end *must* use `energy(calc, at)` but may not call any
#    other variants of `energy`.

"""
* `energy(at, x::Dofs) -> Float64`

`potential_energy`; with potentially added cell terms, e.g., if there is
applied stress or applied pressure.
"""
energy(at::AbstractAtoms, x::Dofs) =
      energy(set_dofs!(at, x))
energy(calc::AbstractCalculator, at::AbstractAtoms, x::Dofs) =
      energy(calc, set_dofs!(at, x))
energy(at::AbstractAtoms) =
      energy(calculator(at), at)

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
gradient(calc::AbstractCalculator, at::AbstractAtoms, x::Dofs) =
      gradient(calc, set_dofs!(at, x))
gradient(at::AbstractAtoms) =
      gradient(calculator(at), at)


"""
`hessian`: compute hessian of total energy with respect to DOFs!

same as gradient, just returns the hessian; new constraints should only
overload `hessian(calc, at, cons)`
"""
function hessian end
hessian(at::AbstractAtoms, x::Dofs) =
      hessian(set_dofs!(at, x))
hessian(calc::AbstractCalculator, at::AbstractAtoms, x::Dofs) =
      hessian(calc, set_dofs!(at, x))
hessian(at::AbstractAtoms) =
      hessian(calculator(at), at)


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
            preconditioner(at, calculator(at))   # should be preconditioner(calc,  at) ???

preconditioner(at::AbstractAtoms, calc::AbstractCalculator) = I



#######################################################################
## Experimental Prototypes for in-place versions
#######################################################################


"""
`energy!`: non-allocating version of `energy`
"""
energy!(calc::AbstractCalculator, at::AbstractAtoms, temp::Nothing) =
      energy(calc, at)

"""
`energy!`: non-allocating version of `forces`
"""
forces!(F::AbstractVector{JVec{T}}, calc::AbstractCalculator,
        at::AbstractAtoms{T}, temp::Nothing) where {T} =
      copy!(F, forces(calc, at))

"""
`energy!`: non-allocating version of `virial`
"""
virial!(calc::AbstractCalculator, at::AbstractAtoms, temp::Nothing) =
      virial(calc, at)
