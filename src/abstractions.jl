
# here we define and document the prototypes that are implemented

"""
`AbstractAtoms`

the abstract supertype for storing atomistic
configurations. A basic implementation might simply store a list of positions.
"""
abstract AbstractAtoms


"""
Abstract supertype of neighbourlists.

The standard contructor should be of the form
```
   NeighbourList(at::AbstractAtoms, rcut)
```
where `rcut` should be either a scalar or a vector of cut-offs.
"""
abstract AbstractNeighbourList

abstract AbstractConstraint

"""
`Dofs`: dof vector
"""
typealias Dofs Vector{Float64}



"""
`AbstractCalculator`: the abstract supertype of calculators. These
store model information, and are linked to the implementation of energy,
forces, and so forth.
"""
abstract AbstractCalculator


"""
`Preconditioner`: abstract base type for preconditioners

Preconditioners need to implement the following three functions:
* `update!(P, at::AbstractAtoms)`
* `A_mul_B!(out::Dof, P, x::Dof)`
* `A_ldiv_B!(out::Dof, P, f::Dof)`
"""
abstract Preconditioner <: AbstractMatrix{Float64}



"""
`macro protofun(fsig::Expr)`

generates "function prototypes" which throw an error message
with a little extra information in case a certain function has not been
implemented. This is in some way duplicating default Julia behaviour, but it
has the advantage that we can write documention for non-existing functions and
it also emphasizes that this specific function is part of the abstract atoms
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
      positions, get_positions, set_positions!, unsafe_positions,
      get_cell, set_cell!, is_cubic, pbc, get_pbc, set_pbc!,
      set_data!, get_data,
      set_calculator!, calculator, get_calculator!,
      set_constraint!, constraint, get_constraint,
      neighbourlist, cutoff,
      stress, site_stresses, site_energies,
      defm, set_defm!

# length is used for several things
import Base: length, A_ldiv_B!, A_mul_B!, cell, gradient

export AbstractCalculator,
      energy, potential_energy, forces, gradient

export AbstractNeighbourList,
       sites, bonds

export AbstractConstraint, NullConstraint, dofs, set_dofs!

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

"return a reference to positions"
@protofun unsafe_positions(::AbstractAtoms)

"Set positions of all atoms as a `3 x N` array."
@protofun set_positions!(::AbstractAtoms, ::JVecs)

set_positions!(at::AbstractAtoms, p::Matrix) = set_positions!(at, vecs(p))

"get computational cell (the rows are the lattice vectors)"
@protofun cell(::AbstractAtoms)

"alias for `cell`"
get_cell = cell

"set computational cell"
@protofun set_cell!(::AbstractAtoms, ::Matrix)

"deformation matrix; `defm(at) = cell(at)'`"
defm(at::AbstractAtoms) = JMat(cell(at)')

"set the deformation matrix"
set_defm!(at::AbstractAtoms, F::JMatF) = set_cell!(at, Matrix(F'))

"set periodic boundary conditions"
@protofun set_pbc!(::AbstractAtoms, ::NTuple{3,Bool})

"get array (or tuple) determining which directions are periodic"
@protofun pbc(::AbstractAtoms)

"alias for `pbc`"
get_pbc = pbc

"determines whether a cubic cell is used (i.e. cell is a diagonal matrix)"
is_cubic(a::AbstractAtoms) = isdiag(cell(a))

"""
`set_data!(at, name, value)`:
associate some data with `at`; to be stored in a Dict within `at`;
if `name` is of type `Symbol` or `String` then can also use
 `setindex!`
"""
@protofun set_data!(a::AbstractAtoms, name::Any, value::Any)
Base.setindex!(at::AbstractAtoms, value,
               name::Union{Symbol, AbstractString}) = set_data!(at, name, value)

"""
`get_data(a, name)`:
obtain some data stored with `set_data!`,
if `name` is of type `Symbol` or `String` then can also use
 `getindex`
"""
@protofun get_data(a::AbstractAtoms, name::Any)
Base.getindex(at::AbstractAtoms,
               name::Union{Symbol, AbstractString}) = get_data(at, name)

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

"""
`neighbourlist(a::AbstractAtoms, rcut)`

construct a suitable neighbourlist for this atoms object. `rcut` should
be allowed to be either a scalar or a vector.
"""
@protofun neighbourlist(a::AbstractAtoms, rcut::Float64)



#######################################################################
#  NEIGHBOURLIST
#######################################################################



"""
`sites(::AbstractNeighbourList)` or `sites(::AbstractAtoms, rcut)`

Returns an iterator over atomic sites.
```julia
for (idx, neigs, r, R, S) in sites(nlist)
    # do something at this site
end
```
Here `idx` is the current center-atom, `neigs` a collection
indexing the neighbouring atoms, `r` a vector if distances and
`R` a vector of distance vectors. `S` stores information
about which copy of the cell the neighbours belong to

A quicker way, if `nlist` won't be reused is
```julia
for n, ... in sites(at, rcut)
```

It should be assumed that `neigs, r, R, S` are views and therefore
should not be modified!
"""
@protofun sites(::AbstractNeighbourList)

sites(at::AbstractAtoms, rcut::Float64) = sites(neighbourlist(at, rcut))


"""
`bonds` : iterator over (pair-) bonds

E.g., the a pair potential can be implemented as follows:
```julia
ϕ(r) = r^(-12) - 2.0 * r^(-6)
dϕ(r) = -12 * (r^(-13) - r^(-7))
function lj(at::AbstractAtoms)
   E = 0.0; dE = zeros(3, length(at)) |> vecs
   for (i, j, r, R, S) in bonds(at, 4.1)
      E += ϕ(r)
      dE[j] += (dϕ(r)/r) * R
      dE[i] -= (dϕ(r)/r) * R
   end
   return E, dE
end
```
"""
@protofun bonds(::AbstractNeighbourList)

bonds(at::AbstractAtoms, cutoff::Float64) = bonds(neighbourlist(at, cutoff))







#######################################################################
#     CALCULATOR
#######################################################################

type  NullCalculator <: AbstractCalculator end

"Returns the cut-off radius of the potential."
@protofun cutoff(::AbstractCalculator)

"""
`energy`: can be called in various ways
*  `energy(calc, at)`: base definition
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
Returns the negative gradient of the total energy in the format.
"""
@protofun forces(c::AbstractCalculator, a::AbstractAtoms)
forces(at::AbstractAtoms) = forces(calculator(at), at)

"`gradient(c,a) = - forces(c, a)`"
gradient(c::AbstractCalculator, a::AbstractAtoms) = scale!(forces(c, a), -1.0)
gradient(at::AbstractAtoms) = gradient(calculator(at), at)

@protofun stress(c::AbstractCalculator, a::AbstractAtoms)
stress(a::AbstractAtoms) = stress(calculator(a), a)

@protofun site_stresses(c::AbstractCalculator, a::AbstractAtoms)
site_stress(a::AbstractAtoms) = site_stress(calculator(a), a)


#######################################################################
#  Constraints
#######################################################################

type NullConstraint <: AbstractConstraint end

"""
`dofs(at::AbstractAtoms, cons::AbstractConstraint) -> Dofs`

Take an atoms object `at` and return a Dof-vector that fully describes the
state given the constraint `cons`
"""
# function dofs end
@protofun dofs(at::AbstractAtoms, cons::AbstractConstraint)

dofs(at::AbstractAtoms) = dofs(at, constraint(at))


"""
`set_dofs!(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs) -> at`

change configuration stored in `at` according to `cons` and `x`.
"""
@protofun set_dofs!(at::AbstractAtoms, cons::AbstractConstraint, x::Dofs)

set_dofs!(at::AbstractAtoms, x::Dofs) = set_dofs!(at, constraint(at), x)


"""
`project!(at::AbstractAtoms, cons::AbstractConstraint) -> at`

project the `at` onto the constraint manifold.
"""
@protofun project!(at::AbstractAtoms, cons::AbstractConstraint)

project!(at::AbstractAtoms) = project!(at, constraint(at))


# converting calculator functionality

energy(at::AbstractAtoms, x::Dofs) = energy(set_dofs!(at, x))

@protofun gradient(at::AbstractAtoms, cons::AbstractConstraint)

gradient(at::AbstractAtoms, x::Dofs) = gradient(set_dofs!(at, x), constraint(at))



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
# TODO: this is a bit of a problem with the nice abstract framework we
#       are constructing here; it can easily happen now that we
#       update positions multiple times

"""
Identity preconditioner, i.e., no preconditioner.
"""
type Identity <: Preconditioner
end

A_ldiv_B!(out::Dofs, P::Identity, x::Dofs) = copy!(out, x)
A_mul_B!(out::Dofs, P::Identity, f::Dofs) = copy!(out, f)
update!(P::Identity, at::AbstractAtoms) = P


"construct a preconditioner suitable for this atoms object"
preconditioner(at::AbstractAtoms) = preconditioner(at, calculator(at), constraint(at))
preconditioner(at::AbstractAtoms, calc::AbstractCalculator, con::AbstractConstraint) =
         Identity()
