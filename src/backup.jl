
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
