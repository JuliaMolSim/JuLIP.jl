
module Chemistry

using JuLIP.FIO: load_dict

export atomic_number,
       chemical_symbol,
       atomic_mass,
       symmetry,
       lattice_constant,
       rnn

data = load_dict(@__DIR__()[1:end-3] * "data/asedata.json")

const _rnn = [Float64(d) for d in data["rnn"]]
const _symbols = [Symbol(s) for s in data["symbols"]]
const _masses = [Float64(m) for m in data["masses"]]
const _refstates = Dict{String, Any}[ d == nothing ? Dict{String, Any}() : d
                                      for d in data["refstates"]]

const _numbers = Dict{Symbol, Int16}()
for (n, sym) in enumerate(_symbols)
   _numbers[sym] = Int16(n - 1)
end

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

lattice_constant(z::Integer) = Float64( _refstates[z+1]["a"] )
lattice_constant(sym::Symbol) = lattice_constant(atomic_number(sym))

refstate(z::Integer) = _refstates[z+1]
refstate(sym::Symbol) = refstate(atomic_number(sym))

end
