using LinearAlgebra: det

# here we define and document the prototypes that are implemented

"""
`AbstractAtoms`

theabstractsupertype for storing atomistic
configurations. A basic implementation might simply store a list of positions.
"""
abstract type AbstractAtoms end


"""
abstract supertype of neighbourlists.

The standard contructor should be of the form
```
   NeighbourList(at::AbstractAtoms, rcut)
```
where `rcut` should be either a scalar or a vector of cut-offs.
"""
abstract type AbstractNeighbourList end

abstract type AbstractConstraint end

"""
`Dofs`: dof vector
"""
const Dofs = Vector{Float64}



"""
`AbstractCalculator`: theabstractsupertype of calculators. These
store model information, and are linked to the implementation of energy,
forces, and so forth.
"""
abstract type AbstractCalculator end


"""
`Preconditioner`:abstractbase type for preconditioners

Preconditioners need to implement the following three functions:
* `update!(P, at::AbstractAtoms)`
* `A_mul_B!(out::Dof, P, x::Dof)`
* `A_ldiv_B!(out::Dof, P, f::Dof)`
"""
abstract type Preconditioner end # <: AbstractMatrix{Float64} end



"""
`macro protofun(fsig::Expr)`

generates "function prototypes" which throw an error message
with a little extra information in case a certain function has not been
implemented. This is in some way duplicating default Julia behaviour, but it
has the advantage that we can write documention for non-existing functions and
it also emphasizes that this specific function is part of theabstractatoms
interface.

## Usage

```julia
"Returns number of atoms"
@protofun length(::AbstractAtoms)
```

This will create a function `length(at::AbstractAtoms)`; if somebody
calls `length(at)` on a concrete sub-type of `AbstractAtoms` for which
`length` hasn't been implemented, it will throw an error.
"""
macro protofun(fsig::Expr)
    @assert fsig.head == :call
    fname = fsig.args[1]
    argnames = Any[]
    for idx in 2:length(fsig.args)
        arg = fsig.args[idx]
        if isa(arg, Expr) && arg.head == :kw
            arg = arg.args[1]
        end
        if isa(arg, Symbol)
            push!(argnames, arg)
        elseif isa(arg, Expr) && arg.head == :(::)
            if length(arg.args) != 2
                @gensym s
                insert!(arg.args, 1, s)
            end
            push!(argnames, arg.args[1])
        end
    end
    body = quote
        error(string("JuLIP: ", $fname,
                     ($([:(typeof($(esc(arg)))) for arg in argnames]...),),
                     " ) has no implementation."))
    end
    return Expr(:function, esc(fsig), body)
end


# function defined primarily on AbstractAtoms
export AbstractAtoms,
      positions, get_positions, set_positions!,
      momenta, get_momenta, set_momenta!,
      masses, get_masses, set_masses,
      cell, get_cell, set_cell!, is_cubic,
      pbc, get_pbc, set_pbc!,
      set_data!, get_data, has_data,
      set_calculator!, calculator, get_calculator,
      set_constraint!, constraint, get_constraint,
      neighbourlist, cutoff,
      defm, set_defm!

# length is used for several things
import Base: length

using LinearAlgebra
import LinearAlgebra: ldiv!, mul!

export AbstractCalculator,
      energy, potential_energy, forces, gradient, hessian,
      site_energies,
      stress, virial, site_virials

export AbstractNeighbourList,
       sites, bonds

export AbstractConstraint, NullConstraint,
         position_dofs, set_position_dofs!,
         momentum_dofs, set_momentum_dofs!,
         dofs, set_dofs!

export Preconditioner, preconditioner

# TODO: probably rename Preconditioner to AbstractPreconditioner and
#       AMGPrecon to Preconditioner
# TODO: iterators for angles, dihedrals


"Return number of atoms"
@protofun length(::AbstractAtoms)

"Return copy of positions of all atoms as a `3 x N` array."
@protofun positions(::AbstractAtoms)

"alias for `positions`"
get_positions = positions

"Set positions of all atoms"
@protofun set_positions!(::AbstractAtoms, ::Any)

set_positions!(at::AbstractAtoms, p::AbstractMatrix) = set_positions!(at, vecs(p))
set_positions!(at::AbstractAtoms, x, y, z) = set_positions!(at, [x'; y'; z'])

xyz(at::AbstractAtoms) = xyz(positions(at))


"Return copy of momenta of all atoms as a `3 x N` array."
@protofun momenta(::AbstractAtoms)

"alias for `momenta`"
get_momenta = momenta

"Set momenta of all atoms as a `3 x N` array."
@protofun set_momenta!(::AbstractAtoms, ::Any)

set_momenta!(at::AbstractAtoms, p::AbstractMatrix) = set_momenta!(at, vecs(p))

@protofun masses(::AbstractAtoms)
get_masses = masses
@protofun set_masses!(::AbstractAtoms, ::Any)

"get computational cell (the rows are the lattice vectors)"
@protofun cell(::AbstractAtoms)

"alias for `cell`"
get_cell = cell

"set computational cell"
@protofun set_cell!(::AbstractAtoms, ::AbstractMatrix)

# TODO: deprecate these!
"deformation matrix; `defm(at) = cell(at)'`"
defm(at::AbstractAtoms) = JMat(cell(at)')

"""
`set_defm!(at::AbstractAtoms, F::AbstractMatrix; updatepositions=false) -> at`

set the deformation matrix
"""
function set_defm!(at::AbstractAtoms, F::AbstractMatrix; updatepositions=false)
   if updatepositions
      A = JMatF(F * inv(defm(at)))
      X = [A * x for x in positions(at)]
      set_positions!(at, X)
   end
   set_cell!(at, Matrix(F'))
end


"set periodic boundary conditions"
@protofun set_pbc!(::AbstractAtoms, ::Union{AbstractVector, Tuple})
set_pbc!(at::AbstractAtoms, p::Bool) = set_pbc!(at, (p,p,p))

"get array (or tuple) determining which directions are periodic"
@protofun pbc(::AbstractAtoms)

"alias for `pbc`"
get_pbc = pbc

"determines whether a cubic cell is used (i.e. cell is a diagonal matrix)"
is_cubic(a::AbstractAtoms) = isdiag(cell(a))


"""
`set_data!(at, name, value)`:
associate some data with `at`; to be stored in a Dict within `at`

if `name::Union{Symbol, AbstractString}`, then `setindex!` is an alias
for `set_data!`.
"""
@protofun set_data!(a::AbstractAtoms, name::Any, value::Any)
Base.setindex!(at::AbstractAtoms, value,
               name::Union{Symbol, AbstractString}) = set_data!(at, name, value)

"""
`get_data(a, name)`:
obtain some data stored with `set_data!`

if `name::Union{Symbol, AbstractString}`, then `getindex` is an alias
for `get_data`.
"""
@protofun get_data(a::AbstractAtoms, name::Any)

Base.getindex(at::AbstractAtoms,
               name::Union{Symbol, AbstractString}) = get_data(at, name)

"check whether some data with id `name` is already stored"
@protofun has_data(a::AbstractAtoms, name::Any)

"delete an atom"
@protofun deleteat!(a::AbstractAtoms, n::Integer)

"alias for `deleteat!`"
delete_atom! = deleteat!

"return an attached calculator"
@protofun calculator(at::AbstractAtoms)

"attach a calculator"
@protofun set_calculator!(at::AbstractAtoms, calc::AbstractCalculator)

"return attached constraint"
@protofun constraint(at::AbstractAtoms)

"attach a constraint"
@protofun set_constraint!(at::AbstractAtoms, cons::AbstractConstraint)



#######################################################################
#  NEIGHBOURLIST
#######################################################################

"""
`neighbourlist(a::AbstractAtoms, rcut)`

construct a suitable neighbourlist for this atoms object. `rcut` should
be allowed to be either a scalar or a vector.
"""
@protofun neighbourlist(a::AbstractAtoms, rcut::AbstractFloat)



# """
# `sites(::AbstractNeighbourList)` or `sites(::AbstractAtoms, rcut)`
#
# Returns an iterator over atomic sites.
# ```julia
# for (idx, neigs, r, R, S) in sites(nlist)
#     # do something at this site
# end
# ```
# Here `idx` is the current center-atom, `neigs` a collection
# indexing the neighbouring atoms, `r` a vector if distances and
# `R` a vector of distance vectors. `S` stores information
# about which copy of the cell the neighbours belong to
#
# A quicker way, if `nlist` won't be reused is
# ```julia
# for (n, ...) in sites(at, rcut)
# ```
#
# It should be assumed that `neigs, r, R, S` are views and therefore
# should not be modified!
# """
# @protofun sites(::AbstractNeighbourList)
#
# sites(at::AbstractAtoms, rcut::Float64) = sites(neighbourlist(at, rcut))
#
#
# """
# `bonds` : iterator over (pair-) bonds
#
# E.g., the a pair potential can be implemented as follows:
# ```julia
# ϕ(r) = r^(-12) - 2.0 * r^(-6)
# dϕ(r) = -12 * (r^(-13) - r^(-7))
# function lj(at::AbstractAtoms)
#    E = 0.0; dE = zeros(3, length(at)) |> vecs
#    for (i, j, r, R, S) in bonds(at, 4.1)
#       E += ϕ(r)
#       dE[j] += (dϕ(r)/r) * R
#       dE[i] -= (dϕ(r)/r) * R
#    end
#    return E, dE
# end
# ```
# """
# @protofun bonds(::AbstractNeighbourList)
#
# bonds(at::AbstractAtoms, cutoff::Float64) = bonds(neighbourlist(at, cutoff))




#######################################################################
#     CALCULATOR
#######################################################################

mutable struct  NullCalculator <: AbstractCalculator end

"Returns the cut-off radius of the potential."
@protofun cutoff(::AbstractCalculator)

cutoff(at::AbstractAtoms) = cutoff(calculator(at))

"""
`energy`: can be called in various ways
*  `energy(calc, at)`: base definition; this is normally the only method that
     needs to be overloaded when a new calculator is implemented.
* `energy(at) = energy(calculator(at), at)`: if a calculator is attached to `at`
* `energy(calc, at, const, dof) = energy(calc, dof2at!(at,const,dof))`
* `energy(calc, at, dof) = energy(calc, at, constraint(dof), dof)`
* `energy(at, dof) = energy`


Return the total potential energy of a configuration of atoms `a`, using the calculator
`c`.
"""
@protofun energy(c::AbstractCalculator, a::AbstractAtoms)
energy(at::AbstractAtoms) = energy(calculator(at), at)

"`potential_energy` : alias for `energy`"
potential_energy = energy


@protofun site_energies(c::AbstractCalculator, a::AbstractAtoms)
site_energies(a::AbstractAtoms) = site_energies(calculator(a), a)

# """
# energy difference between two configurations; default is to just compute the
# two energies, but this allows implementation of numerically robust
# differences,w hich can be important for very large problems.
# """
# energy_difference(c::AbstractCalculator, a::AbstractAtoms, aref::AbstractAtoms) =
#    energy(c, a) - energy(c, aref)

"""
forces in `Vector{JVecF}`  (negative gradient w.r.t. atom positions only)
"""
@protofun forces(c::AbstractCalculator, a::AbstractAtoms)
forces(at::AbstractAtoms) = forces(calculator(at), at)

"""
hessian with respect to all atom positions
"""
@protofun hessian_pos(calc::AbstractCalculator, at::AbstractAtoms)
hessian_pos(at::AbstractAtoms) = hessian_pos(calculator(at), at)

"""
* `virial(c::AbstractCalculator, a::AbstractAtoms) -> JMatF`
* `virial(a::AbstractAtoms) -> JMatF`

returns virial, (- ∂E / ∂F) where `F = defm(a)`
"""
@protofun virial(c::AbstractCalculator, a::AbstractAtoms)

virial(a::AbstractAtoms) = virial(calculator(a), a)

"""
* `stress(c::AbstractCalculator, a::AbstractAtoms) -> JMatF`
* `stress(a::AbstractAtoms) -> JMatF`

stress = - virial / volume; this function should *not* be overloaded;
instead overload virial
"""
stress(c::AbstractCalculator, a::AbstractAtoms) = - virial(c, a) / det(defm(a))

stress(at::AbstractAtoms) = stress(calculator(at), at)

# remove these for now; not clear they are useful.
# @protofun site_virials(c::AbstractCalculator, a::AbstractAtoms)
# site_virials(a::AbstractAtoms) = site_virials(calculator(a), a)


#######################################################################
#  Constraints and DoF Handling
#######################################################################

mutable struct NullConstraint <: AbstractConstraint end

"""
`position_dofs(at::AbstractAtoms, cons::AbstractConstraint) -> Dofs`
`position_dofs(at::AbstractAtoms) -> Dofs`
`dofs(at::AbstractAtoms, cons::AbstractConstraint) -> Dofs`

Take an atoms object `at` and return a Dof-vector that fully describes the
state given the constraint `cons`
"""
@protofun position_dofs(at::AbstractAtoms, cons::AbstractConstraint)
position_dofs(at::AbstractAtoms) = position_dofs(at, constraint(at))

"""
`dofs` is an alias for `positions_dofs`
"""
dofs(args...) = position_dofs(args...)

"""
* `set_position_dofs!(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs) -> at`
* `set_position_dofs!(at::AbstractAtoms, x::Dofs) -> at`
* `set_dofs!(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs) -> at`
* `set_dofs!(at::AbstractAtoms, cons::AbstractConstraint) -> at`

change configuration stored in `at` according to `cons` and `x`.
"""
@protofun set_position_dofs!(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs)
set_position_dofs!(at::AbstractAtoms, x::Dofs) = set_position_dofs!(at, constraint(at), x)
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
@protofun momentum_dofs(at::AbstractAtoms, cons::AbstractConstraint)
momentum_dofs(at::AbstractAtoms) = momentum_dofs(at, constraint(at))


"""
* `set_momentum_dofs!(at::AbstractAtoms, cons::AbstractConstraint, p::Dofs) -> at`
* `set_momentum_dofs!(at::AbstractAtoms, p::Dofs) -> at`

change configuration stored in `at` according to `cons` and `p`.
"""
@protofun set_momentum_dofs!(at::AbstractAtoms, cons::AbstractConstraint, q::Dofs)
set_momentum_dofs!(at::AbstractAtoms, q::Dofs) = set_momentum_dofs!(at, constraint(at), q)

"""
* `project!(at::AbstractAtoms, cons::AbstractConstraint) -> at`
* `project!(at::AbstractAtoms) -> at`

project the `at` onto the constraint manifold.
"""
@protofun project!(at::AbstractAtoms, cons::AbstractConstraint)

project!(at::AbstractAtoms) = project!(at, constraint(at))


# converting calculator functionality

"""
* `gradient(at, cons::AbstractConstraint) -> Float64`
* `gradient(at, x::Dofs) -> Float64`

`potential_energy`; with potentially added cell terms, e.g., if there is
applied stress or applied pressure.
"""
@protofun energy(at::AbstractAtoms, cons::AbstractConstraint)

energy(at::AbstractAtoms, x::Dofs) = energy(set_dofs!(at, x), constraint(at))


"""
* `gradient(at) -> Dofs`
* `gradient(at, cons::AbstractConstraint) -> Dofs`
* `gradient(at, x::Dofs) -> Dofs`

gradient of `potential_energy` in dof-format;
depending on the constraint this could include e.g. a gradient w.r.t. cell
shape
"""
@protofun gradient(at::AbstractAtoms, cons::AbstractConstraint)

gradient(at::AbstractAtoms, x::Dofs) = gradient(set_dofs!(at, x), constraint(at))

gradient(at::AbstractAtoms) = gradient(at, constraint(at))

"""
`hessian`: compute hessian of total energy with respect to DOFs!

TODO: write docs
"""
@protofun hessian(at::AbstractAtoms, cons::AbstractConstraint)

hessian(at::AbstractAtoms, x::Dofs) = hessian(set_dofs!(at, x), constraint(at))
hessian(at::AbstractAtoms) = hessian(at, constraint(at))


#######################################################################
#                     PRECONDITIONER
#######################################################################


"""
`update!(precond::Preconditioner, at::AbstractAtoms)`

Update the preconditioner with the new geometry information.
"""
@protofun update!(precond::Preconditioner, at::AbstractAtoms)
update!(precond::Preconditioner, at::AbstractAtoms, x::Dofs) =
            update!(precond, set_dofs!(at, x))
# TODO: this is a bit of a problem with the niceabstractframework we
#       are constructing here; it can easily happen now that we
#       update positions multiple times

"""
Identity preconditioner, i.e., no preconditioner.
"""
mutable struct Identity <: Preconditioner
end

ldiv!(out::Dofs, P::Identity, x::Dofs) = copyto!(out, x)
mul!(out::Dofs, P::Identity, f::Dofs) = copyto!(out, f)
update!(P::Identity, at::AbstractAtoms) = P


"construct a preconditioner suitable for this atoms object"
preconditioner(at::AbstractAtoms) = preconditioner(at, calculator(at), constraint(at))
preconditioner(at::AbstractAtoms, calc::AbstractCalculator, con::AbstractConstraint) =
         Identity()
