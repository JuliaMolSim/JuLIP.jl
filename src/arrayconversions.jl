


import Base.convert

export mat, vecs
export SVec, SMat, STen, JVec, JVecs, JVecF, JVecsF
export JMat, JMatF
export zerovecs, maxdist, maxnorm

"typealias fora static vector type; currently `StaticArrays.SVector`"
const SVec = SVector
"typealias fora static matrix type; currently `StaticArrays.SMatrix`"
const SMat = SMatrix
"typealias fora static array type; currently `StaticArrays.SArray`"
const STen = SArray

"`JVec{T}` : 3-dimensional immutable vector"
const JVec{T} = SVec{3,T}
const JVecF = JVec{Float64}
const JVecI = JVec{Int}

# Base.zero{T}(::Type{JVec{T}}) = JVec([zero(T) for i=1:3])

# which of these types shall we deprecate?

"`JVecs{T}` : List of 3-dimensional immutable vectors"
const JVecs{T} = Vector{JVec{T}}
const JVecsF = JVecs{Float64}
const JVecsI = JVecs{Int}

"""
`vecs(V::Matrix)` : convert (as reference) a 3 x N matrix representing
N vectors (e.g. forces) in R³ to a list (vector) of fixed-size-array vectors.

`vecs(V::Vector)` : assumes that V is morally a 3 x N matrix, stored in a long
vector

`vecs(V::Array{T,N})` : If `V` has dimensions 3 x n2 x ... x nN then
it gets converted to an n2 x ... x nN array with JVec{T} entries.
"""
vecs(V::Matrix{T}) where {T} = (@assert size(V,1) == 3;
                                reinterpret(JVec{T}, vec(V)))
vecs(V::Vector{T}) where {T} = reinterpret(JVec{T}, V)
vecs(V::AbstractArray{T,N}) where {T,N} =
      reshape( reinterpret(JVec{T}, vec(V)), tuple(size(V)[2:end]...) )
# TODO: figure out how to convert back by useinf .parent.parent


"`JMat{T}` : 3 × 3 immutable matrix"
const JMat{T,N} = SMatrix{3,3,T,N}
const JMatF = SMatrix{3,3,Float64,9}
const JMatI = JMat{Int}

# Base.zero{T}(::Type{JMat{T}}) = JMat([zero(T) for i = 1:9])
# Base.eye{T}(::Type{JMat{T}}) = JMat(T, eye(3))

"`JMats{T}` : (2-dimensional) Array of 3 × 3 immutable matrices"
const JMats{T} = Array{JMat{T}}
const JMatsF = JMats{Float64}
const JMatsI = JMats{Int}
#
# """
# `mats(V::Matrix)` : convert (as reference) a 3 x 3 x N x N tensor representing
# N × N matrices (e.g. local hess) in R³ to a list (vector) of fixed-size-array vectors.
#
# `mats(V::Vector)` : assumes that V is morally a list of  x N matrix, stored in a long
# vector
#
# `mats(V::Array{T,N})` : If `V` has dimensions 3 x n2 x ... x nN then
# it gets converted to an n2 x ... x nN array with JVec{T} entries.
# """
# mats{T}(V::Array{T}) = reinterpret(JMat{T}, V, tuple(size(V)[3:end]...))
# mats{T}(V::Vector{T}) = reinterpret(JVec{T}, V, (length(V) ÷ 3,)
# mats{T,N}(V::Array{T,N}) = reinterpret(JMat{T}, V, tuple(size(V)[3:end]...))
# mats{T}(V::Array{T}) = vecs(vecs(V))
# mats{T}(V::Matrix{T}) = reshape(permutedims(V, [1 3 2 4]), 6, 6)

export mats
function mats(V::Array{T, N}) where {T,N}
  @assert size(V,1) == size(V,2) == 3
  @assert ndims(V) > 2
  return reinterpret(JMat{T}, V, tuple(size(V)[3:end]...))
end

export xyz

"""
convert a Vector{JVec{T}} or a 3 x N Matrix{T} into a 3-tuple
(x, y, z). E.g.,
```
x, y, z = positions(at) |> xyz
```
a short-cut is
```
x, y, z = xyz(at)
```
Conversely, `set_positions!(at, x, y, z)` is also allowed.
"""
xyz(V::Vector{JVec{T}}) where {T} = xyz(mat(V))


function xyz(V::Matrix)
   @assert size(V, 1) == 3
   return (view(V, 1, :), view(V, 2, :), view(V, 3, :))
end



"""
`mat(X::JVecs)`: convert (as reference) a list (Vector) of
fixed-size-array points or vecs to a 3 x N matrix representing those.

### Usage:
```
X = positions(at)          # returns a Vector{JVec}
X = positions(at) |> mat   # returns a Matrix
```
"""
mat(V::Vector{SVec{N,T}}) where {N,T} = reshape( reinterpret(T, V), (N, length(V)) )
mat(X::AbstractVector{SVec{N,T}}) where {N,T} = mat(collect(X))
mat(X::Base.ReinterpretArray) = reshape(X.parent, 3, :)

# rewrite all of this in terms of `convert` (TODO: is this needed?)
convert(::Type{Matrix{T}}, V::JVecs{T}) where {T} = mat(V)
convert(::Type{JVecs{T}}, V::Matrix{T}) where {T} = vec(V)

# initialise a vector of vecs or points
zerovecs(n::Integer) = zerovecs(Float64, n)
zerovecs(T::Type, n::Integer) = zeros(T, 3, n) |> vecs

# TODO: delete this!
# zeromats(T::Type, n::Integer) = zeros()
# # initialise a matrix of

"""
maximum of distances between two sets of JVec's, usually positions;
`maximum(a - b for (a,b) in zip(x,y))`
"""
function maxdist(x::AbstractArray{JVec{T}}, y::AbstractArray{JVec{T}}) where T
   @assert length(x) == length(y)
   return maximum( norm(a-b)  for (a,b) in zip(x,y) )
end

"`maximum(norm(y) for y in x);` typically, x is a vector of forces"
maxnorm(X::AbstractVector{JVec{T}}) where {T} = maximum( norm(x) for x in X )
