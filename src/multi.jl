using JuLIP.Chemistry: atomic_number
using JuLIP: Atoms
using StaticArrays: SVector

abstract type MSitePotential <: AbstractCalculator end

abstract type MPairPotential <: MSitePotential end

# Experimental Multi-component codes

# ----------------------------------------------------------------------
#    Managing a list of species
# ----------------------------------------------------------------------

abstract type AbstractZList end

"""
`ZList` and `SZList{NZ}` : simple data structures that store a list
of species and convert between atomic numbers and the index in the list.
Can be constructed via
* `ZList(zors)` : where `zors` is an Integer  or `Symbol` (single species)
* `ZList(zs1, zs2, ..., zsn)`
* `ZList([sz1, zs2, ..., zsn])`
* All of these take a kwarg `static = {true, false}`; if `true`, then `ZList`
will return a `SZList{NZ}` for (possibly) faster access.
"""
struct ZList <: AbstractZList
   list::Vector{Int16}
end

Base.length(zlist::AbstractZList) = length(zlist.list)

struct SZList{N} <: AbstractZList
   list::SVector{N, Int16}
end

ZList(zlist::AbstractVector{<: Integer}; static = false) = (
   static ? SZList(SVector( (Int16.(sort(zlist)))... ))
          :  ZList( convert(Vector{Int16}, sort(zlist)) ))

ZList(s::Symbol; kwargs...) =
      ZList( [ atomic_number(s) ]; kwargs... )

ZList(S::AbstractVector{Symbol}; kwargs...) =
      ZList( atomic_number.(S); kwargs... )

ZList(args...; kwargs) =
      ZList( [args...]; kwargs...)


i2z(Zs::AbstractZList, i::Integer) = Zs.list[i]

function z2i(Zs::AbstractZList, z::Integer)
   for j = 1:length(Zs.list)
      if Zs.list[j] == z
         return j
      end
   end
   error("z = $z not found in ZList $(Zs.list)")
end

i2z(V, i::Integer) = i2z(V.zlist, i)
z2i(V, z::Integer) = z2i(V.zlist, z)

Dict(zlist::ZList) = Dict("__id__" => "JuLIP_ZList",
                          "list" => zlist.list)
Base.convert(::Val{:JuLIP_ZList}, D::Dict) = ZList(D)
ZList(D::Dict) = ZList(D["list"])

Dict(zlist::SZList) = Dict("__id__" => "JuLIP_SZList",
                          "list" => zlist.list)
Base.convert(::Val{:JuLIP_SZList}, D::Dict) = SZList(D)
SZList(D::Dict) = SZList(SVector(Int16.(D["list"])...))


# ----------------------------------------------------------------------
#       Basic calculator extensions for multi-species
# ----------------------------------------------------------------------


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

function neigsz(nlist::PairList, at::Atoms, i::Integer)
   j, R = NeighbourLists.neigs(nlist, i)
   return j, R, at.Z[j]
end




alloc_temp(V::MSitePotential, at::AbstractAtoms) =
      alloc_temp(V, maxneigs(neighbourlist(at, cutoff(V))))

alloc_temp(V::MSitePotential, N::Integer) =
      ( R = zeros(JVecF, N),
        Z = zeros(Int16, N), )

alloc_temp_d(V::MSitePotential, at::AbstractAtoms) =
      alloc_temp_d(V, maxneigs(neighbourlist(at, cutoff(V))))

alloc_temp_d(V::MSitePotential, N::Integer) =
      (dV = zeros(JVecF, N),
        R = zeros(JVecF, N),
        Z = zeros(Int16, N), )

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
      evaluate_d!(tmp.dV, tmp, calc, R, Z, Int16(at.Z[i]))
      for a = 1:length(j)
         frc[j[a]] -= tmp.dV[a]
         frc[i]    += tmp.dV[a]
      end
   end
   return frc
end

function virial!(tmp, calc::MSitePotential, at::Atoms{T};
                 domain=1:length(at)) where {T}
   nlist = neighbourlist(at, cutoff(calc))
   vir = zero(JMat{T})
   for i in domain
      j, R, Z = neigsz!(tmp, nlist, at, i)
      evaluate_d!(tmp.dV, tmp, calc, R, Z, Int16(at.Z[i]))
      vir += site_virial(tmp.dV, R)
   end
   return vir
end


evaluate(V::MSitePotential, R, Z, z) =
      evaluate!(alloc_temp(V, length(R)), V, R, Z, z)

evaluate_d(V::MSitePotential, R::AbstractVector{JVec{T}}, Z, z) where {T} =
      evaluate_d!(zeros(JVec{T}, length(R)),
                  alloc_temp_d(V, length(R)),
                  V, R, Z, z)


site_energy(V::MSitePotential, at::AbstractAtoms, i0::Integer) =
      energy(V, at; domain = (i0,))

site_energy_d(V::MSitePotential, at::AbstractAtoms, i0::Integer) =
      rmul!(forces(V, at; domain = (i0,)), -one(eltype(at)))

# -------- Dispatch for MPairPotentials ------------------------------------

evaluate!(tmp, V::MPairPotential, R::AbstractVector, Z::AbstractVector, z0) =
   sum( evaluate!(tmp, V, norm(R[i]), Z[i], z0)
        for i = 1:length(R) )

function evaluate_d!(dV, tmp, V::MPairPotential, R, Z, z0)
   dV = tmp.dV
   for i = 1:length(R)
      r = norm(R[i])
      dV[i] = (evaluate_d!(tmp, V, r, Z[i], z0)/r) * R[i]
   end
   return dV
end



# -----------------------------------------------------------------------------






# struct MPairPotential <: MSitePotential
#    Z2V::Dict{Tuple{Int16,Int16}, WrappedPairPotential}
# end
#
# MPairPotential() =
#       MPairPotential(Dict{(Int16, Int16), WrappedPairPotential}())
#
# function MPairPotential(args...)
#    Vm = MPairPotential()
#    for p in args
#       @assert p isa Pair
#       z, V = _convert_multi_pp(sym::Symbol, p)
#       Vm.Z2V[z] = V
#    end
#    return Vm
# end
#
# F64fun(V::F64fun) = V
# _convert_multi_pp(sym::Tuple{Symbol, Symbol}, V) =
#       _convert_multi_pp(atomic_number.(sym), V)
# _convert_multi_pp(z::Tuple{<:Integer,<:Integer}, V) =
#       Int16.(z), V
#
# cutoff(V::MPairPotential) =
#    maximum( cutoff(V) for (key, V) in V.Z2V )
