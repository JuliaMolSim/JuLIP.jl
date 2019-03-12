
import Base: convert, ==

# import JuLIP: energy, virial, site_energies, forces
# import JuLIP.Potentials: evaluate

using JuLIP: JVec, JMat

export OneBody

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

site_energies(V::OneBody, at::Atoms) = fill(V(), length(at))

energy(V::OneBody, at::Atoms) = length(at) * V()

forces(V::OneBody, at::Atoms{T}) where {T} = zeros(JVec{T}, length(at))

virial(V::OneBody, at::Atoms{T}) where {T} = zero(JMat{T})

Dict(V::OneBody) = Dict("__id__" => "OneBody", "E0" => V.E0)

OneBody(D::Dict) = OneBody(D["E0"])

convert(::Val{:OneBody}, D::Dict) = OneBody(D)

==(V1::OneBody, V2::OneBody) = (V1.E0 == V2.E0)
