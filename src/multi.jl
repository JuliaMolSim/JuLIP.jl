using JuLIP.Chemistry: atomic_number
using JuLIP: Atoms

abstract type MultiSitePotential end

# Experimental Multi-component codes

"""
`neigsz!(tmp, nlist::PairList, at::Atoms, i::Integer) -> j, R Z`

requires a temporary storage array `tmp` with fields
`tmp.R, tmp.Z`.
"""
function neigsz!(tmp, nlist::PairList, at::Atoms, i::Integer)
   j, R = neigs!(tmp.R, nlist, i)
   Z = tmp.Z
   for n = 1:length(j)
      Z[n] = at.Z[j[n]]
   end
   return j, R, (@view Z[1:length(j)])
end


struct MultiPairPotential <: MultiSitePotential
   Z2V::Dict{Tuple{Int16,Int16}, WrappedPairPotential}
end

MultiPairPotential() =
      MultiPairPotential(Dict{(Int16, Int16), WrappedPairPotential}())

function MultiPairPotential(args...)
   Vm = MultiPairPotential()
   for p in args
      @assert p isa Pair
      z, V = _convert_multi_pp(sym::Symbol, p)
      Vm.Z2V[z] = V
   end
   return Vm
end

F64fun(V::F64fun) = V
_convert_multi_pp(sym::Tuple{Symbol, Symbol}, V) =
      _convert_multi_pp(atomic_number.(sym), V)
_convert_multi_pp(z::Tuple{<:Integer,<:Integer}, V) =
      Int16.(z), V

cutoff(V::MultiPairPotential) =
   maximum( cutoff(V) for (key, V) in V.Z2V )



alloc_temp(V::MultiSitePotential, at::AbstractAtoms) =
      alloc_temp(V, maxneigs(neighbourlist(at, cutoff(V))))

alloc_temp(V::MultiSitePotential, N::Integer) =
      ( R = zeros(JVecF, N),
        Z = zeros(JVec{Int16}, N), )

alloc_temp_d(V::MultiSitePotential, at::AbstractAtoms) =
      alloc_temp_d(V, maxneigs(neighbourlist(at, cutoff(V))))

alloc_temp_d(V::MultiSitePotential, N::Integer) =
      (dV = zeros(JVecF, N),
        R = zeros(JVecF, N),
        Z = zeros(JVec{Int16}, N), )

energy(V::MultiSitePotential, at::AbstractAtoms; kwargs...) =
      energy!(alloc_temp(V, at), V, at; kwargs...)

virial(V::MultiSitePotential, at::AbstractAtoms; kwargs...) =
      virial!(alloc_temp_d(V, at), V, at; kwargs...)

forces(V::MultiSitePotential, at::AbstractAtoms{T}; kwargs...) where {T} =
      forces!(zeros(JVec{T}, length(at)), alloc_temp_d(V, at), V, at; kwargs...)


function energy!(tmp, calc::MultiSitePotential, at::Atoms{T}) where {T}
   E = 0.0
   nlist = neighbourlist(at, cutoff(calc))
   for i = 1:length(at)
      j, R, Z = neigsz!(tmp, nlist, at, i)
      E += evaluate!(tmp, calc, R, Z)
   end
   return E
end
