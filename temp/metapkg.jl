
module M

mutable struct Atoms
   X
   P
   M
end

for (S, name) in zip( ["X", "P", "M"], ["positions", "momenta", "masses"] )
for (S, name) in zip( [:X, :P, :M], [:positions, :momenta, :masses] )
   get_name = "get_" * name

   eval( :(
      function get_$name(at::Atoms, $S)
         at.$S = $S
         return at
      end
      ) )
end

end


mutable struct T
   X
end
S = :X
name = "positions"
set_name = parse("set_$name")


@eval begin
   function $set_name(at::T, $S)
      at.$S = $S
      return at
   end
end

t = T([])

set_positions(t, [1])

t
