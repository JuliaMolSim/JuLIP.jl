
import Base: convert, ==

# import JuLIP: energy, virial, site_energies, forces
# import JuLIP.Potentials: evaluate

import JuLIP.FIO: read_dict, write_dict
using JuLIP: JVec, JMat, chemical_symbols

export OneBody


"""
`mutable struct OneBody{T}  <: NBodyFunction{1}`

this should not normally be constructed by a user, but instead E0 should be
passed to the relevant lsq functions, which will construct it.
This structure deals with multi-species configurations.
"""
mutable struct OneBody{T <: AbstractFloat} <: AbstractCalculator
   E0::Dict{Symbol, T}
end

@pot OneBody

OneBody(args...) = OneBody(Dict(args...))


evaluate(V::OneBody, sp) = V.E0[sp]

function site_energies(V::OneBody, at::AbstractAtoms{T}) where {T}
   E = zeros(T, length(at))
   for i in 1:length(at)
      E[i] = V(chemical_symbols(at)[i])
   end
   return E
end

energy(V::OneBody, at::AbstractAtoms) = sum(site_energies(V, at))

forces(V::OneBody, at::AbstractAtoms{T}) where {T} = zeros(JVec{T}, length(at))

virial(V::OneBody, at::AbstractAtoms{T}) where {T} = zero(JMat{T})

write_dict(V::OneBody) =
         Dict("__id__" => "JuLIP_OneBody",
              "E0" => Dict([String(key) => val for (key, val) in V.E0]...))

# convert the E0 Dict{String} read from JSON into a Dict{Symbol}
OneBody(D::Dict{String}, T=Float64) =
   OneBody([Symbol(key) => T(val) for (key, val) in D]...)

read_dict(::Val{:JuLIP_OneBody}, D::Dict) = OneBody(D["E0"])

==(V1::OneBody, V2::OneBody) = (V1.E0 == V2.E0)
