
module Chemistry

using JSON

export atomic_number,
       chemical_symbol,
       atomic_mass,
       symmetry,
       a,
       rnn,
       jbulk

data = JSON.parsefile(@__FILE__()[1:end-16] * "data/asedata.json")

_symbols = [Symbol(s) for s in data["symbols"]]
_masses = [Float64(m) for m in data["masses"]]
_refstates = Dict{String, Any}[ d == nothing ? Dict{String, Any}() : d
                                for d in data["refstates"]]

# the next array is generated from bulk data. I don't like storing it
# like this at all, this should be rewritten, but I don't fell like it
# right now.
const _rnn = [-1.0, -1.0, -1.0, 3.02243, 2.22873, -1.0, 1.54586, -1.0, -1.0,
   -1.0, 3.13248, 3.66329, 3.19823, 2.86378, 2.35126, -1.0, -1.0, -1.0, 3.71938,
   4.52931, 3.94566, 3.25752, 2.89607, 2.6154, 2.49415, -1.0, 2.48549, 2.49875,
   2.48902, 2.55266, 2.66, -1.0, 2.45085, -1.0, 3.53122, -1.0, 4.04465, 4.84108,
   4.29921, 3.55822, 3.17748, 2.85788, 2.72798, 2.70767, 2.64627, 2.68701,
   2.75065, 2.89207, 2.98, -1.0, -1.0, -1.0, 3.91893, -1.0, 4.38406, 5.23945,
   4.34745, 3.72861, 3.64867, 3.6416, 3.63168, -1.0, -1.0, 3.99238, 3.57345,
   3.524, 3.50263, 3.48854, 3.46905, 3.44956, 3.88202, 3.44157, 3.13374,
   2.86654, 2.73664, 2.73976, 2.67994, 2.71529, 2.77186, 2.885, -1.0, 3.41215,
   3.50018, -1.0, 3.35, -1.0, -1.0, -1.0, -1.0, 3.75474, 3.5921, -1.0, -1.0,
   -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
   -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,-1.0, -1.0, -1.0]

const _numbers = Dict{Symbol, Int}()
for (n, sym) in enumerate(_symbols)
   _numbers[sym] = n - 1
end


_unit_cells = Dict(     # (positions, cell matrix, factor of a)
   :fcc => ( [ [0.0,0.0,0.0], ],
             [0 1 1; 1 0 1; 1 1 0],  0.5),
   :bcc => ( [ [0.0,0.0,0.0], ],
             [-1 1 1; 1 -1 1; 1 1 -1], 0.5),
   :diamond => ( [ [0.0, 0.0, 0.0], [0.5, 0.5, 0.5] ],
                 [0 1 1; 1 0 1; 1 1 0], 0.5)
)

_cubic_cells = Dict(   # (positions, factor of a)
   :fcc => ( [ [0 0 0], [0 1 1], [1 0 1], [1 1 0] ], 0.5 ),
   :bcc => ( [ [0 0 0], [1 1 1] ], 0.5 ),
   :diamond => ( [ [0 0 0], [1 1 1], [0 2 2], [1 3 3], [2 0 2],
                   [3 1 3], [2 2 0], [3 3 1] ], 0.25 )
)


atomic_number(sym::Symbol) = _numbers[sym]

chemical_symbol(z::Integer) = _symbols[z+1]

atomic_mass(z::Integer) = _masses[z+1]
atomic_mass(sym::Symbol) = atomic_mass(atomic_number(sym))

element_name(z::Integer) = _names[z+1]
element_name(sym::Symbol) = element_name(atomic_number(sym))

rnn(z::Integer) = _rnn[z+1]       # >>> TODO: move to utils ?? bulk as well??
rnn(sym::Symbol) = rnn(atomic_number(sym))

symmetry(z::Integer) = Symbol(_refstates[z+1]["symmetry"])
symmetry(sym::Symbol) = symmetry(atomic_number(sym))

a(z::Integer) = Float64( _refstates[z+1]["a"] )
a(sym::Symbol) = a(atomic_number(sym))


end
