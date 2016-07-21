
# here we define and document the prototypes that are implemented, e.g.,

import Base.length


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
`macro protofun(fsig::Expr)`

generates "function prototypes" which throw an error message
with a little extra information in case a certain function has not been
implemented. This is in some way duplicating default Julia behaviour, but it
has the advantage that we can write documention for non-existing functions and
it also emphasizes that this specific function is part of the abstract atoms
interface.

## Usage

```jl
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
                     " ) has no implementation.") )
    end
    Expr(:function, esc(fsig), body)
end

# function defined primarily on AbstractAtoms
export AbstractAtoms,
      positions, get_positions, set_positions!,
      cell, get_cell, set_cell!, is_cubic, pbc, get_pbc, set_pbc!,
      set_data!, get_data,
      calculator, get_calculator!, constraint, get_constraint!,
      neighbourlist

# length is used for several things
import Base.length

export AbstractCalculator,
      energy, potential_energy, forces, grad

export AbstractNeighbourList,
      sites, bonds

export AbstractConstraint, NullConstraint, dofs

# TODO: iterator for angles, dihedrals
# TODO: decide on maxforce



# the following are dummy method definitions that just throw an error if a
# method hasn't been implemented

"Return number of atoms"
@protofun length(::AbstractAtoms)

"Return reference to positions of all atoms as a `3 x N` array."
@protofun positions(::AbstractAtoms)

"alias for `positions`"
get_positions = positions

"Set positions of all atoms as a `3 x N` array."
@protofun set_positions!(::AbstractAtoms, ::JPts)

set_positions!(at::AbstractAtoms, p::Matrix) = set_positions!(at, pts(p))

"get computational cell"
@protofun cell(::AbstractAtoms)

"alias for `cell`"
get_cell = cell

"set computational cell"
@protofun set_cell!(::AbstractAtoms, ::Matrix)

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
# energy() TODO: CONTINUE HERE

"`potential_energy` : alias for `energy`"
potential_energy = energy

"""
energy difference between two configurations; default is to just compute the
two energies, but this allows implementation of numerically robust
differences,w hich can be important for very large problems.
"""
energy_difference(c::AbstractCalculator, a::AbstractAtoms, aref::AbstractAtoms) =
   energy(c, a) - energy(c, aref)


"""
Returns the negative gradient of the total energy in the format `3 x length`.
"""
@protofun forces(c::AbstractCalculator, a::AbstractAtoms)

"`grad(c,a) = - forces(c, a)`"
grad(c::AbstractCalculator, a::AbstractAtoms) = - forces(c, a)




#######################################################################
#  Constraints
#######################################################################

type NullConstraint <: AbstractConstraint end

"""
* `dofs(cons::AbstractConstraint, vecs::JVecs)`
* `dofs(cons::AbstractConstraint, vecs::JVecs)`

Take a direction in position space (e.g. a collection of forces)
and project it to a dof-vector
"""
@protofun dofs(at::AbstractAtoms, cons::AbstractConstraint, v_or_p)
dofs(at::AbstractAtoms, cons::AbstractConstraint) = dofs(at, cons, positions(at))
dofs(at::AbstractAtoms) = dofs(at, constraint(at))


"""
`vecs(cons::AbstractConstraint, vecs::JVecs)`:

Take a dof-vector and reconstruct a direction in position space
"""
@protofun vecs(cons::AbstractConstraint, at::AbstractAtoms, dofs::Dofs)

@protofun positions(cons::AbstractConstraint, at::AbstractAtoms, dofs::Dofs)

set_positions!(cons::AbstractConstraint, at::AbstractAtoms, dofs::Dofs) =
      set_positions!(at, positions(cons, at, dofs))


"""
`project(cons::AbstractConstraint, at::AbstractAtoms)`

take an atomistic configuration (collection of positions) and project
onto the manifold defined by the constraint.
"""
@protofun project!(at::AbstractAtoms, cons::AbstractConstraint)



#######################################################################
#  TODO: PRECONDITIONER
#######################################################################

# ==================================================
# more abstract types that eventually need to be
# arranged in a better way

# abstract Preconditioner

# """`max_force(f::AbstractArray{2})`:
#
# For a 3 x N array `f`, return the maximum of `|f[:,n]|₂`.
# """
# maxforce(f) = sqrt(maximum(sumabs2(f, 1)))


# ==================================================
# do we want this at all?

# """
# Returns `(neigs, r, R)` where `neigs` is an integer vector with indices
# of neighbour atoms of `n`, `r` a Float vector with their distances and
# `R` a vector of distance *vectors*.
# """
# @protofun neighbours(n::Integer, neigs::AbstractNeighbourList,
#                          atm::AbstractAtoms; rcut=-1)
#
#
# get_neighbours(n::Integer, neigs::AbstractNeighbourList,
#            atm::AbstractAtoms; rcut=-1) =
#                neighbours(n, neigs, atm; rcut=rcut)
