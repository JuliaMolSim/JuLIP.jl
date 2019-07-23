using JuLIP.Chemistry: atomic_number
using JuLIP: Atoms

abstract type MSitePotential end

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


struct MPairPotential <: MSitePotential
   Z2V::Dict{Tuple{Int16,Int16}, WrappedPairPotential}
end

MPairPotential() =
      MPairPotential(Dict{(Int16, Int16), WrappedPairPotential}())

function MPairPotential(args...)
   Vm = MPairPotential()
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

cutoff(V::MPairPotential) =
   maximum( cutoff(V) for (key, V) in V.Z2V )



alloc_temp(V::MSitePotential, at::AbstractAtoms) =
      alloc_temp(V, maxneigs(neighbourlist(at, cutoff(V))))

alloc_temp(V::MSitePotential, N::Integer) =
      ( R = zeros(JVecF, N),
        Z = zeros(JVec{Int16}, N), )

alloc_temp_d(V::MSitePotential, at::AbstractAtoms) =
      alloc_temp_d(V, maxneigs(neighbourlist(at, cutoff(V))))

alloc_temp_d(V::MSitePotential, N::Integer) =
      (dV = zeros(JVecF, N),
        R = zeros(JVecF, N),
        Z = zeros(JVec{Int16}, N), )

energy(V::MSitePotential, at::AbstractAtoms; kwargs...) =
      energy!(alloc_temp(V, at), V, at; kwargs...)

virial(V::MSitePotential, at::AbstractAtoms; kwargs...) =
      virial!(alloc_temp_d(V, at), V, at; kwargs...)

forces(V::MSitePotential, at::AbstractAtoms{T}; kwargs...) where {T} =
      forces!(zeros(JVec{T}, length(at)), alloc_temp_d(V, at), V, at; kwargs...)


function energy!(tmp, calc::MSitePotential, at::Atoms{T};
                 domain=1:length(at)) where {T}
   E = 0.0
   nlist = neighbourlist(at, cutoff(calc))
   for i in domain
      j, R, Z = neigsz!(tmp, nlist, at, i)
      E += evaluate!(tmp, calc, R, Z, Int16(at.Z[i]))
   end
   return E
end

function forces!(frc, tmp, calc::MSitePotential, at::Atoms{T};
                 domain=1:length(at), reset=true) where {T}
   if reset; fill!(frc, zero(JVec{T})); end
   nlist = neighbourlist(at, cutoff(calc))
   for i in domain
      j, R, Z = neigsz!(tmp, nlist, at, i)
      evaluate_d!(tmp.dV, tmp, V, R, Z, Int16(at.Z[i]))
      for a = 1:length(j)
         frc[j[a]] -= tmp.dV[a]
         frc[i]    += tmp.dV[a]
      end
   end
   return frc
end
