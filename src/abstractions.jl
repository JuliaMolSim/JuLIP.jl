
# here we define and document the prototypes that are implemented, e.g.,

import Base.length

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
      neighbourlist

# length is used for several things
import Base.length

export AbstractCalculator,
      energy, potential_energy, forces, grad

export AbstractNeighbourList,
      sites, bonds, angles, dihedrals

# TODO: angles, dihedrals
# TODO: decide on maxforce


"""
`AbstractAtoms`

the abstract supertype for storing atomistic
configurations. A basic implementation might simply store a list of positions.
"""
abstract AbstractAtoms

# the following are dummy method definitions that just throw an error if a
# method hasn't been implemented



"Return number of atoms"
@protofun length(::AbstractAtoms)

"Return reference to positions of all atoms as a `3 x N` array."
@protofun positions(::AbstractAtoms)

"alias for `positions`"
get_positions(a::AbstractAtoms) = positions(a)

"Set positions of all atoms as a `3 x N` array."
@protofun set_positions!(::AbstractAtoms, ::JPts)

set_positions!(at::AbstractAtoms, p::Matrix) = set_positions!(at, pts(p))

"get computational cell"
@protofun cell(::AbstractAtoms)

"alias for `cell`"
get_cell(a::AbstractAtoms) = cell(a)

"set computational cell"
@protofun set_cell!(::AbstractAtoms, ::Matrix)

"set periodic boundary conditions"
@protofun set_pbc!(::AbstractAtoms, ::NTuple{3,Bool})

"get array (or tuple) determining which directions are periodic"
@protofun pbc(::AbstractAtoms)

"alias for `pbc`"
get_pbc(a::AbstractAtoms) = pbc(a)

"determines whether a cubic cell is used (i.e. cell is a diagonal matrix)"
is_cubic(a::AbstractAtoms) = isdiag(cell(a))

"associate some data with `a`; to be stored in a Dict within `a`"
@protofun set_data!(a::AbstractAtoms, name::Any, value::Any)

"obtain some data stored with `set_data!`"
@protofun get_data(a::AbstractAtoms, name::Any)

"delete an atom"
@protofun deleteat!(a::AbstractAtoms, n::Integer)

"alias for `deleteat!`"
delete_atom!(a::AbstractAtoms, n::Integer) = deleteat!(a, n)


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
Abstract supertype of neighbourlists.

The standard contructor should be of the form
```
   NeighbourList(at::AbstractAtoms, rcut)
```
where `rcut` should be either a scalar or a vector of cut-offs.
"""
abstract AbstractNeighbourList


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





#######################################################################
#     CALCULATOR
#######################################################################


"""
`AbstractCalculator`: the abstract supertype of calculators. These
store model information, and are linked to the implementation of energy,
forces, and so forth.
"""
abstract AbstractCalculator

"Returns the cut-off radius of the potential."
@protofun cutoff(::AbstractCalculator)

"""
Return the total energy of a configuration of atoms `a`, using the calculator
`c`.
"""
@protofun energy(c::AbstractCalculator, a::AbstractAtoms)

"`potential_energy` : alias for `energy`"
potential_energy(c::AbstractCalculator, a::AbstractAtoms) = energy(c, a)

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
#  TODO: CONSTRAINTS
#######################################################################

# # constraints implement boundary conditions, or other types
# # of constraints; the details of this interface are still a bit fuzzy for me

# """Abstract supertype for constraints; these are objects that implement boundary
# conditions."""
# abstract AbstractConstraints

# "Return constraints attached to the atoms object (if one exists)"
# @protofun get_constraints(::AbstractAtoms)

# "Attach a constraints object to the atoms object"
# @protofun set_constraints!(::AbstractAtoms, ::AbstractConstraints)


# """Returns a bare `Vector{T <: FloatingPoint}` object containing the degrees of
# freedom describing the state of the simulation. This function should be
# overloaded for concrete implementions of `AbstractAtoms` and
# `AbstractConstraints`.

# Alternative wrapper function
#     get_dofs(atm::AbstractAtoms) = get_dofs(atm, get_constraints(atm))
# """
# @protofun get_dofs(a::AbstractAtoms, c::AbstractConstraints)
# get_dofs(atm::AbstractAtoms) = get_dofs(atm, get_constraints(atm))


# """Takes a \"dual\" array (3 x lenght) and applies the dual constraints
# to obtain effective forces acting on dofs. Returns a vector of the same
# length as dofs."""
# @protofun forces_to_dofs{T <: AbstractFloat}(f::Matrix{T}, con::AbstractConstraints)



# ==================================================
# more abstract types that eventually need to be
# arranged in a better way

# abstract Preconditioner

# """`max_force(f::AbstractArray{2})`:
#
# For a 3 x N array `f`, return the maximum of `|f[:,n]|₂`.
# """
# maxforce(f) = sqrt(maximum(sumabs2(f, 1)))
