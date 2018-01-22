# TODO: rename `unsafe_***` >>> `***_ref`



using Parameters
export Atoms


"""
`JData`: datatype for storing any data

some data which needs to be updated if the configuration (positions only!) has
changed too much.
"""
mutable struct JData{T}
   max_change::T     # how much X may change before recomputing
   accum_change::T   # how much has it changed already
   data::Any
end


"""
`Atoms{T <: AbstractFloat} <: AbstractAtoms`

The
"""
@with_kw mutable struct Atoms{T <: AbstractFloat} <: AbstractAtoms
   X::JVecs{T} = JVecs{T}[]   # positions
   P::JVecs{T} = JVecs{T}[]   # momenta (or velocities?)
   M::JVecs{T} = JVecs{T}[]   # masses
   cell::JMat{T} = zero(JMat{T})                  # cell
   pbc::NTuple{3, Bool} = (false, false, false)   # boundary condition
   calc::AbstractCalculator = NullCalculator()
   cons::AbstractConstraint = NullConstraint()
   data::Dict{Any,JData{T}} = Dict{Any,JData{T}}()
end

Atoms() = Atoms{Float64}()

# derived properties
length(at::Atoms) = length(at.X)


# access to struct fields
# ------------------------
symbols = (:X, :P, :M, :cell, :calc, :cons, :pbc)
names = ("positions", "momenta", "masses", "cell", "calculator", "constraint", "pbc")

for (S, name) in zip(symbols, names)
   set_name = parse("set_$(name)!")
   get_name = parse("get_$name")
   unsafe_set_name = parse("unsafe_set_$(name)!")
   unsafe_get_name = parse("unsafe_get_$name")
   @eval begin
      function $(get_name)(at::Atoms)
         return copy(at.$S)
      end
      function $(unsafe_get_name)(at::Atoms)
         return at.$S
      end
      function $(set_name)(at::Atoms, Q)
         if length(at.$S) != length(Q)
            at.$S = copy(Q)
         else
            at.$S .= Q
         end
         return at
      end
      function $(unsafe_set_name)(at::Atoms, Q)
         at.$S = Q
         return at
      end
   end
end

# an alias for positions
Base.getindex(at::Atoms, i::Integer) = at.X[i]
function Base.setindex!(at::Atoms{T}, i::Integer, x::JVec) where T <: AbstractFloat
   at.X[i] = JVec{T}(x)
   return at.X[i]
end

# access to data fields
# ----------------------


      cell, get_cell, set_cell!, is_cubic, pbc, get_pbc, set_pbc!,
      # set_data!, get_data, has_data,
      set_calculator!, calculator, get_calculator!,
      set_constraint!, constraint, get_constraint,
      neighbourlist, cutoff,
      defm, set_defm!
