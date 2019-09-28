
import Base: convert, ==

# import JuLIP: energy, virial, site_energies, forces
# import JuLIP.Potentials: evaluate

using JuLIP: JVec, JMat, chemical_symbols

export OneBody, MOneBody

"""
`mutable struct OneBody{T}  <: NBodyFunction{1}`

this should not normally be constructed by a user, but instead E0 should be
passed to the relevant lsq functions, which will construct it.
"""
mutable struct OneBody{T} <: AbstractCalculator
   E0::T
end

@pot OneBody


evaluate(V::OneBody) = V.E0

site_energies(V::OneBody, at::AbstractAtoms) = fill(V(), length(at))

energy(V::OneBody, at::AbstractAtoms; domain = 1:length(at)) = length(domain) * V()
forces(V::OneBody, at::AbstractAtoms{T}; kwargs...) where {T} = zeros(JVec{T}, length(at))
virial(V::OneBody, at::AbstractAtoms{T}; kwargs...) where {T} = zero(JMat{T})
site_energy(V::OneBody, at::AbstractAtoms, i0::Integer) = V()
site_energy_d(V::OneBody, at::AbstractAtoms{T}, i0::Integer) where {T} =
      zeros(JVec{T}, length(at))

Dict(V::OneBody) = Dict("__id__" => "OneBody", "E0" => V.E0)
OneBody(D::Dict) = OneBody(D["E0"])
convert(::Val{:OneBody}, D::Dict) = OneBody(D)

==(V1::OneBody, V2::OneBody) = (V1.E0 == V2.E0)

import Base: *
*(c::Real, V::OneBody) = OneBody(V.E0 * c)
*(V::OneBody, c::Real) = c * V

"""
`mutable struct MOneBody{T}  <: NBodyFunction{1}`

this should not normally be constructed by a user, but instead E0 should be
passed to the relevant lsq functions, which will construct it.
This structure deals with multi-species configurations.
"""
mutable struct MOneBody{T <: AbstractFloat} <: AbstractCalculator
   E0::Dict{Symbol, T}
end

@pot MOneBody


evaluate(V::MOneBody, sp) = V.E0[sp]

function site_energies(V::MOneBody, at::AbstractAtoms{T}) where {T}
   E = zeros(T, length(at))
   for i in 1:length(at)
      E[i] = V(chemical_symbols(at)[i])
   end
   return E
end

energy(V::MOneBody, at::AbstractAtoms) = sum(site_energies(V, at))

forces(V::MOneBody, at::AbstractAtoms{T}) where {T} = zeros(JVec{T}, length(at))

virial(V::MOneBody, at::AbstractAtoms{T}) where {T} = zero(JMat{T})

Dict(V::MOneBody) = Dict("__id__" => "MOneBody", "E0" => V.E0)

convert_str_2_symb(D::Dict{Symbol,T}) where {T} = D #already in the correct form

# convert the E0 Dict{String} read from JSON into a Dict{Symbol}
MOneBody(D::Dict{String}, T=Float64) =
   Dict( Symbol(key) => T(val) for (key, val) in D ) |> MOneBody

convert(::Val{:MOneBody}, D::Dict) = MOneBody(D["E0"])

==(V1::MOneBody, V2::MOneBody) = (V1.E0 == V2.E0)
