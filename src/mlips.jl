
"""
Prototype for IPs based on "Machine-Learning", to be imported by
modules that either define basis sets or regression methods
"""
module MLIPs

using JuLIP:       AbstractCalculator, AbstractAtoms
using JuLIP.FIO:   decode_dict

import JuLIP:      energy, forces, virial, site_energy, site_energy_d
import Base:       Dict, convert, ==

export IPSuperBasis, IPCollection, combine

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
abstract type IPBasis end

# ========== wrap one more more calculators into a basis =================

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

Dict(coll::IPCollection) = Dict(
      "__id__" => "JuLIP_IPCollection",
      "coll" => Dict.(coll.coll) )
IPCollection(D::Dict) = IPCollection( decode_dict.( D["coll"] ) )
convert(::Val{:JuLIP_IPCollection}, D::Dict) = IPCollection(D)
import Base.==
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

Dict(superB::IPSuperBasis) = Dict(
      "__id__" => "JuLIP_IPSuperBasis",
      "components" => Dict.(superB.BB) )
IPSuperBasis(D::Dict) = IPSuperBasis( decode_dict.( D["components"] ) )
convert(::Val{:JuLIP_IPSuperBasis}, D::Dict) = IPSuperBasis(D)
import Base.==
==(B1::IPSuperBasis, B2::IPSuperBasis) = all(B1.BB .== B2.BB)

# ========== SumIP =================
# a sum of several IPs.

struct SumIP{T} <: AbstractCalculator
   components::Vector{T}
end

SumIP(args...) = SumIP([args...])

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


Dict(sumip::SumIP) = Dict(
      "__id__" => "JuLIP_SumIP",
      "components" => Dict.(sumip.components) )
SumIP(D::Dict) = SumIP( decode_dict.( D["components"] ) )
convert(::Val{:JuLIP_SumIP}, D::Dict) = SumIP(D)

SumIP(V::AbstractCalculator, sumip::SumIP) =
   SumIP( [ [V]; sumip.components ]  )
SumIP(sumip::SumIP, V::AbstractCalculator) =
   SumIP( [ sumip.components; [V] ]  )
SumIP(sum1::SumIP, sum2::SumIP) =
   SumIP( [ sum1.components; sum2.components ]  )

end
