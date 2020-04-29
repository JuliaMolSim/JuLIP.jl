
module Chemistry

using JuLIP.FIO: load_dict

export atomic_number,
       chemical_symbol,
       atomic_mass,
       symmetry,
       lattice_constant,
       rnn,
       AtomicNumber

"""
`struct AtomicNumber` : wrapper type to encode atomic numbers. Makes it
easier to dispatch on `AtomicNumber` and not confuse with a species index in
a list e.g.
"""
struct AtomicNumber <: Unsigned
   z::UInt16
end

AtomicNumber(z::AtomicNumber) = z
(TI::Type{<: Integer})(z::AtomicNumber) = TI(z.z)
Base.convert(TI::Type{<: Integer}, z::AtomicNumber) = convert(TI, z.z)
Base.hash(z::AtomicNumber, h::UInt) = hash(z.z, h)
Base.show(io::IO, z::AtomicNumber) = print(io, "<$(z.z)>")

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

atomic_number(sym::Symbol) = AtomicNumber(_numbers[sym])

chemical_symbol(z::AtomicNumber) = _symbols[z.z+1]

atomic_mass(z::AtomicNumber) = _masses[z.z+1]
atomic_mass(sym::Symbol) = atomic_mass(atomic_number(sym))

element_name(z::AtomicNumber) = _names[z.z+1]
element_name(sym::Symbol) = element_name(atomic_number(sym))

rnn(z::AtomicNumber) = _rnn[z.z+1]
rnn(sym::Symbol) = rnn(atomic_number(sym))

symmetry(z::AtomicNumber) = Symbol(_refstates[z.z+1]["symmetry"])
symmetry(sym::Symbol) = symmetry(atomic_number(sym))

lattice_constant(z::AtomicNumber) = Float64( _refstates[z.z+1]["a"] )
lattice_constant(sym::Symbol) = lattice_constant(atomic_number(sym))

refstate(z::AtomicNumber) = _refstates[z.z+1]
refstate(sym::Symbol) = refstate(atomic_number(sym))

end
