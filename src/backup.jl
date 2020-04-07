
#        field symbol   name        is array
fields = [ (:X,       "positions",    true),
           (:P,       "momenta",      true),
           (:M,       "masses",       true),
           (:Z,       "numbers",      true),
           (:cell,    "cell",         false),
           (:calc,    "calculator",   false),
           (:cons,    "constraint",   false),
           (:pbc,     "pbc",          false),
           ]

for (S, name, isarray) in fields
   set_name = Meta.parse("set_$(name)!")
   get_name = Meta.parse("$name")
   if isarray
      @eval begin
         function $(get_name)(at::Atoms)
            return copy(at.$S)
         end
         function $(set_name)(at::Atoms, Q)
            if length(at.$S) != length(Q)
               at.$S = copy(Q)
            else
               at.$S .= Q
            end
            return at
         end
      end
   else
      function $(get_name)(at::Atoms)
         return at.$S
      end
      function $(set_name)(at::Atoms, Q)
         at.$S = Q
         return at
      end
   end
end



# # an FF preconditioner for pair potentials
# function precon!(tmp, V::PairPotential, r::T, R::JVec{T}, innerstab=T(0.1)
#                  ) where {T <: Number}
#    r = norm(R)
#    dV = evaluate_d!(tmp, V, r)
#    ddV = evaluate_dd!(tmp, V, r)
#    R̂ = R/r
#    return (1-innerstab) * (abs(ddV) * R̂ * R̂' + abs(dV / r) * (I - R̂ * R̂')) +
#              innerstab  * (abs(ddV) + abs(dV / r)) * I
# end


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
