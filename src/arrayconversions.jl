

using StaticArrays, PyCall

import Base.convert

export mat, vecs
export SVec, SMat, STen, JVec, JVecs, JVecF, JVecsF
export JMat, JMatF
export zerovecs, maxdist, maxnorm

export unsafe_pyarrayref, safe_pyarrayref

"typealias for a static vector type; currently `StaticArrays.SVector`"
typealias SVec SVector
"typealias for a static matrix type; currently `StaticArrays.SMatrix`"
typealias SMat SMatrix
"typealias for a static array type; currently `StaticArrays.SArray`"
typealias STen SArray

"`JVec{T}` : 3-dimensional immutable vector"
typealias JVec{T} SVec{3,T}
typealias JVecF JVec{Float64}
typealias JVecI JVec{Int}

Base.zero{T}(::Type{JVec{T}}) = JVec([zero(T) for i=1:3])

"`JVecs{T}` : List of 3-dimensional immutable vectors"
typealias JVecs{T} Vector{JVec{T}}
typealias JVecsF JVecs{Float64}
typealias JVecsI JVecs{Int}

"""
`vecs(V::Matrix)` : convert (as reference) a 3 x N matrix representing
N vectors (e.g. forces) in R³ to a list (vector) of fixed-size-array vectors.

`vecs(V::Vector)` : assumes that V is morally a 3 x N matrix, stored in a long
vector

`vecs(V::Array{T,N})` : If `V` has dimensions 3 x n2 x ... x nN then
it gets converted to an n2 x ... x nN array with JVec{T} entries.
"""
vecs{T}(V::Matrix{T}) = reinterpret(JVec{T}, V, (size(V,2),))
vecs{T}(V::Vector{T}) = reinterpret(JVec{T}, V, (length(V) ÷ 3,))
vecs{T,N}(V::Array{T,N}) = reinterpret(JVec{T}, V, tuple(size(V)[2:end]...))

"`JMat{T}` : 3 × 3 immutable marix"

typealias JMat SMatrix{3,3}
typealias JMatF JMat{Float64}
typealias JMatI JMat{Int}

Base.zero{T}(::Type{JMat{T}}) = JMat([zero(T) for i = 1:9])
Base.eye{T}(::Type{JMat{T}}) = JMat(T, eye(3))

"`JMats{T}` : (2-dimensional) Array of 3 × 3 immutable matrices"
typealias JMats{T} Array{JMat{T}}
typealias JMatsF JMats{Float64}
typealias JMatsI JMats{Int}

"""
`mats(V::Matrix)` : convert (as reference) a 3 x 3 x N x N tensor representing
N × N matrices (e.g. local hess) in R³ to a list (vector) of fixed-size-array vectors.

`mats(V::Vector)` : assumes that V is morally a list of  x N matrix, stored in a long
vector

`mats(V::Array{T,N})` : If `V` has dimensions 3 x n2 x ... x nN then
it gets converted to an n2 x ... x nN array with JVec{T} entries.
"""
"mats{T}(V::Array{T}) = reinterpret(JMat{T}, V, tuple(size(V)[3:end]...))
mats{T}(V::Vector{T}) = reinterpret(JVec{T}, V, (length(V) ÷ 3,)
mats{T,N}(V::Array{T,N}) = reinterpret(JMat{T}, V, tuple(size(V)[3:end]...))"
mats{T}(V::Array{T}) = vecs(vecs(V))
mats{T}(V::Matrix{T}) = reshape(permutedims(V, [1 3 2 4]), 6, 6)

"""
`mat(X::JVecs)`: convert (as reference) a list (Vector) of
fixed-size-array points or vecs to a 3 x N matrix representing those.

### Usage:
```
X = positions(at)          # returns a Vector{JVec}
X = positions(at) |> mat   # returns a Matrix
```
"""
mat{N,T}(V::Vector{SVec{N,T}}) = reinterpret(T, V, (N, length(V)))
mat{N,T}(X::AbstractVector{SVec{N,T}}) = mat(collect(X))

# rewrite all of this in terms of `convert` (TODO: is this needed?)
convert{T}(::Type{Matrix{T}}, V::JVecs{T}) = mat(V)
convert{T}(::Type{JVecs{T}}, V::Matrix{T}) = vec(V)

# initialise a vector of vecs or points
zerovecs(n::Integer) = zerovecs(Float64, n)
zerovecs(T::Type, n::Integer) = zeros(T, 3, n) |> vecs

zeromats(T::Type, n::Integer) = zeros()
# initialise a matrix of

"""
maximum of distances between two sets of JVec's, usually positions;
`maximum(a - b for (a,b) in zip(x,y))`
"""
function maxdist{T}(x::JVecs{T}, y::JVecs{T})
   @assert length(x) == length(y)
   return maximum( norm(a-b)  for (a,b) in zip(x,y) )
end

"`maximum(norm(y) for y in x);` typically, x is a vector of forces"
maxnorm{T}(x::JVecs{T}) = maximum( norm.(x) )



# The next function is conversion of python arrays  _by reference_
pyarrayref(a::PyObject; own=false) = pyarrayref(PyArray(a); own=own)
pyarrayref(a::PyArray; own=false) = unsafe_wrap(Array, a.data, reverse(a.dims), own)
