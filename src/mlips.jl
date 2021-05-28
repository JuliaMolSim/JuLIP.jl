
"""
Prototype for IPs based on "Machine-Learning", to be imported by
modules that either define basis sets or regression methods
"""
module MLIPs

using JuLIP:       AbstractCalculator, AbstractAtoms, JVec, AtomicNumber, JMat

import JuLIP:      energy, forces, virial, site_energy, site_energy_d,
                   alloc_temp, alloc_temp_d, evaluate, evaluate_d,
                   evaluate!, evaluate_d!, evaluate_ed,
                   read_dict, write_dict, fltype, rfltype

import JuLIP.Potentials: site_virial

import ACEbase: combine

import Base:       ==

export IPSuperBasis, IPCollection, combine 

abstract type AbstractBasis <: AbstractCalculator end


"""
`abstract type IPBasis` : A type derived from `IPBasis` defines
not an IP as such but a collection of IPs that can be understood as
a basis set. E.g., a 2B-potential could be obtained from
polynomials,
```
   ϕ(r) = ∑_k c_k exp(-k r)
```
then the `B::IPBasis` might specify the basis functions {r -> exp(-kr)} as
a collection of `PairPotential`s.

Then, `energy(at, B)` does not return a single energy but a vector
of energies.

## Developer Notes

* `length(basis::IPBasis)` must return the number of basis functions
"""
abstract type IPBasis <: AbstractBasis end

"""
`alloc_B(B, x)`

if `B::IPBasis` and `x` is some argument, then allocate storage to evaluate
the basis when evaluated with argument `x`.
"""
alloc_B(basis, args...) = zeros(fltype(basis), length(basis))

"""
`alloc_dB(B, x)`

if `B::IPBasis` and `x` is some argument, then allocate storage to evaluate
the derivative of the basis when evaluated with argument `x`.
"""
alloc_dB(basis, x::AbstractVector, args...) = alloc_dB(basis, length(x))
alloc_dB(basis, x::Union{Number, JVec}, args...) = alloc_dB(basis)
alloc_dB(basis) = zeros(JVec{fltype(basis)}, length(basis))
alloc_dB(basis, N::Integer) = zeros(JVec{fltype(basis)}, (length(basis), N))

evaluate(B::IPBasis, x, args...) =
   evaluate!(alloc_B(B, x), alloc_temp(B, x), B, x, args...)

function evaluate_ed(B::IPBasis, x, args...)
   b = alloc_B(B, x)
   db = alloc_dB(B, x)
   tmp = alloc_temp_d(B, x)
   evaluate_d!(b, db, tmp, B, x, args...)
   return b, db
end

evaluate_d(B::IPBasis, x, args...) = evaluate_ed(B::IPBasis, x, args...)[2]


# ========== wrap one more more calculators into a basis =================

# the main difference with IPSuperBasis is that an IPCollection is
# a basis of individual calculators, whereas an IPSuperBasis combines
# two or more `IPBasis` sets.

struct IPCollection{T} <: IPBasis
   coll::Vector{T}
end

IPCollection(args...) = IPCollection([args...])
Base.length(coll::IPCollection) = length(coll.coll)
combine(basis::IPCollection, coeffs::AbstractVector{<:Number}) =
      SumIP([ c * b for (c, b) in zip(coeffs, basis.coll) ])

energy(coll::IPCollection, at::AbstractAtoms) =
         [ energy(B, at) for B in coll.coll ]
forces(coll::IPCollection, at::AbstractAtoms) =
         [ forces(B, at) for B in coll.coll ]
virial(coll::IPCollection, at::AbstractAtoms) =
         [ virial(B, at) for B in coll.coll ]
site_energy(coll::IPCollection, at::AbstractAtoms, i0::Integer) =
         [ site_energy(V, at, i0) for V in coll.coll ]
site_energy_d(coll::IPCollection, at::AbstractAtoms, i0::Integer) =
         [ site_energy_d(V, at, i0) for V in coll.coll ]

write_dict(coll::IPCollection) = Dict(
            "__id__" => "JuLIP_IPCollection",
            "coll" => Dict.(coll.coll) )
IPCollection(D::Dict) = IPCollection( read_dict.( D["coll"] ) )
read_dict(::Val{:JuLIP_IPCollection}, D::Dict) = IPCollection(D)
==(B1::IPCollection, B2::IPCollection) = all(B1.coll .== B2.coll)


# ========== IPSuperBasis : combine multiple sub-basis =================


"""
`struct IPSuperBasis:` a collection of IP basis sets, re-interpreted
as a large basis
"""
struct IPSuperBasis{TB <: IPBasis} <: IPBasis
   BB::Vector{TB}
end

_convert_basis_(b::IPBasis) = b
_convert_basis_(b::AbstractCalculator) = IPCollection(b)
_convert_basis_(b::AbstractVector) = IPCollection(b)

IPSuperBasis(args...) = IPSuperBasis([_convert_basis_(b) for b in args])

Base.length(super::IPSuperBasis) = sum(length.(super.BB))

function combine(basis::IPSuperBasis, coeffs::AbstractVector{<:Number})
   i0 = 0
   components = []
   for B in basis.BB
      lenB = length(B)
      push!(components, combine(B, coeffs[i0+1:i0+lenB]))
      i0 += lenB
   end
   return SumIP(components)
end

energy(superB::IPSuperBasis, at::AbstractAtoms) =
         vcat([ energy(B, at) for B in superB.BB ]...)
forces(superB::IPSuperBasis, at::AbstractAtoms) =
         vcat([ forces(B, at) for B in superB.BB ]...)
virial(superB::IPSuperBasis, at::AbstractAtoms) =
         vcat([ virial(B, at) for B in superB.BB ]...)
site_energy(superB::IPSuperBasis, at::AbstractAtoms, i0::Integer) =
         vcat([ site_energy(B, at, i0) for B in superB.BB ]...)
site_energy_d(superB::IPSuperBasis, at::AbstractAtoms, i0::Integer) =
         vcat([ site_energy_d(B, at, i0) for B in superB.BB ]...)

write_dict(superB::IPSuperBasis) = Dict(
      "__id__" => "JuLIP_IPSuperBasis",
      "components" => write_dict.(superB.BB) )
IPSuperBasis(D::Dict) = IPSuperBasis( read_dict.( D["components"] ) )
read_dict(::Val{:JuLIP_IPSuperBasis}, D::Dict) = IPSuperBasis(D)
==(B1::IPSuperBasis, B2::IPSuperBasis) = all(B1.BB .== B2.BB)

# ========== SumIP =================
# a sum of several IPs.

struct SumIP{T} <: AbstractCalculator
   components::Vector{T}
end

SumIP(args...) = SumIP([args...])

==(V1::SumIP, V2::SumIP) = all(V1.components .== V2.components)

energy(sumip::SumIP, at::AbstractAtoms; kwargs...) =
         sum(energy(calc, at; kwargs...) for calc in sumip.components)
forces(sumip::SumIP, at::AbstractAtoms; kwargs...) =
         sum(forces(calc, at; kwargs...) for calc in sumip.components)
virial(sumip::SumIP, at::AbstractAtoms; kwargs...) =
         sum(virial(calc, at; kwargs...) for calc in sumip.components)
site_energy(sumip::SumIP, at::AbstractAtoms, i0::Integer) =
         sum(site_energy(V, at, i0) for V in sumip.components)
site_energy_d(sumip::SumIP, at::AbstractAtoms, i0::Integer) =
         sum(site_energy_d(V, at, i0) for V in sumip.components)


write_dict(sumip::SumIP) = Dict(
      "__id__" => "JuLIP_SumIP",
      "components" => write_dict.(sumip.components) )
SumIP(D::Dict) = SumIP( read_dict.( D["components"] ) )
read_dict(::Val{:JuLIP_SumIP}, D::Dict) = SumIP(D)

SumIP(V::AbstractCalculator, sumip::SumIP) =
   SumIP( [ [V]; sumip.components ]  )
SumIP(sumip::SumIP, V::AbstractCalculator) =
   SumIP( [ sumip.components; [V] ]  )
SumIP(sum1::SumIP, sum2::SumIP) =
   SumIP( [ sum1.components; sum2.components ]  )




# -------------------------------------------------------------
#       JuLIP Calculator functionality for IPBasis
# -------------------------------------------------------------

using NeighbourLists: maxneigs
using JuLIP: sites, neighbourlist, cutoff, JVec
using JuLIP.Potentials: neigsz!

function energy(shipB::IPBasis, at::AbstractAtoms{T}) where {T}
   E = zeros(fltype(shipB), length(shipB))
   B = alloc_B(shipB)
   nlist = neighbourlist(at, cutoff(shipB); storelist=false)
   maxnR = maxneigs(nlist)
   tmp = alloc_temp(shipB, maxnR)
   tmpRZ = (R = zeros(JVec{T}, maxnR), Z = zeros(AtomicNumber, maxnR))
   for i = 1:length(at)
      j, R, Z = neigsz!(tmpRZ, nlist, at, i)
      fill!(B, 0)
      evaluate!(B, tmp, shipB, R, Z, at.Z[i])
      E[:] .+= B[:]
   end
   return E
end


function forces(shipB::IPBasis, at::AbstractAtoms{T}) where {T}
   # precompute the neighbourlist to count the number of neighbours
   nlist = neighbourlist(at, cutoff(shipB); storelist=false)
   maxR = maxneigs(nlist)
   # allocate space accordingly
   F = zeros(JVec{T}, length(at), length(shipB))
   B = alloc_B(shipB, maxR)
   dB = alloc_dB(shipB, maxR)
   tmp = alloc_temp_d(shipB, maxR)
   tmpRZ = (R = zeros(JVec{T}, maxR), Z = zeros(AtomicNumber, maxR))
   return forces_inner!(shipB, at, nlist, F, B, dB, tmp, tmpRZ)
end

# this is a little hack to remove a type instability. It probably makes no
# difference in practise...
function forces_inner!(shipB::IPBasis, at::AbstractAtoms{T},
                       nlist, F, B, dB, tmp, tmpRZ) where {T}
   # assemble site gradients and write into F
   for i = 1:length(at)
      j, R, Z = neigsz!(tmpRZ, nlist, at, i)
      fill!(dB, zero(JVec{T}))
      fill!(B, 0)
      evaluate_d!(dB, tmp, shipB, R, Z, at.Z[i])
      for a = 1:length(R)
         F[j[a], :] .-= dB[:, a]
         F[i, :] .+= dB[:, a]
      end
   end
   return [ F[:, iB] for iB = 1:length(shipB) ]
end



function virial(shipB::IPBasis, at::AbstractAtoms{T}) where {T}
   # precompute the neighbourlist to count the number of neighbours
   nlist = neighbourlist(at, cutoff(shipB); storelist=false)
   maxR = maxneigs(nlist)
   # allocate space accordingly
   V = zeros(JMat{T}, length(shipB))
   B = alloc_B(shipB, maxR)
   dB = alloc_dB(shipB, maxR)
   tmp = alloc_temp_d(shipB, maxR)
   tmpRZ = (R = zeros(JVec{T}, maxR), Z = zeros(AtomicNumber, maxR))
   # assemble site gradients and write into F
   for i = 1:length(at)
      j, R, Z = neigsz!(tmpRZ, nlist, at, i)
      fill!(dB, zero(JVec{T}))
      evaluate_d!(dB, tmp, shipB, R, Z, at.Z[i])
      for iB = 1:length(shipB)
         V[iB] += site_virial(dB[iB, :], R)
      end
   end
   return V
end


function _get_neigs(at::AbstractAtoms{T}, i0::Integer, rcut) where {T}
   nlist = neighbourlist(at, rcut)
   maxR = maxneigs(nlist)
   tmpRZ = (R = zeros(JVec{T}, maxR), Z = zeros(AtomicNumber, maxR))
   j, R, Z = neigsz!(tmpRZ, nlist, at, i0)
   return j, R, Z
end

function site_energy(basis::IPBasis, at::AbstractAtoms, i0::Integer)
   j, Rs, Zs = _get_neigs(at, i0, cutoff(basis))
   return evaluate(basis, Rs, Zs, at.Z[i0])
end


function site_energy_d(basis::IPBasis, at::AbstractAtoms{T}, i0::Integer) where {T}
   Ineigs, Rs, Zs = _get_neigs(at, i0, cutoff(basis))
   dEs = [ zeros(JVec{T}, length(at)) for _ = 1:length(basis) ]
   dB = alloc_dB(basis, length(Rs))
   tmp = alloc_temp_d(basis, length(Rs))
   evaluate_d!(dB, tmp, basis, Rs, Zs, at.Z[i0])
   @assert dB isa Matrix{JVec{T}}
   @assert size(dB) == (length(Rs), length(basis))
   for iB = 1:length(basis), n = 1:length(Ineigs)
      dEs[iB][Ineigs[n]] += dB[n, iB]
      dEs[iB][i0] -= dB[n, iB]
   end
   return dEs
end


end
