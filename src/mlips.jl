
"""
Prototype for IPs based on "Machine-Learning", to be imported by
modules that either define basis sets or regression methods
"""
module MLIPs

using JuLIP:      AbstractCalculator, AbstractAtoms
using JuLIP.FIO:  decode_dict

import JuLIP:      energy, forces, virial
import Base:       Dict, convert, ==

export IPSuperBasis, combine

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

"""
`struct IPSuperBasis:` a collection of IP basis sets, re-interpreted
as a large basis
"""
struct IPSuperBasis{TB <: IPBasis} <: IPBasis
   BB::Vector{TB}
end

IPSuperBasis(args...) = IPSuperBasis([args...])

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

energy(sumip::SumIP, at::AbstractAtoms) =
         sum(energy(calc, at) for calc in sumip.components)
forces(sumip::SumIP, at::AbstractAtoms) =
         sum(forces(calc, at) for calc in sumip.components)
virial(sumip::SumIP, at::AbstractAtoms) =
         sum(virial(calc, at) for calc in sumip.components)

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
