
using StaticArrays
using Base: ReinterpretArray, ReshapedArray
using LinearAlgebra: norm

import Base.convert

export mat, vecs, xyz
export SVec, SMat, JVec, JVecF, JMat, JMatF
export maxdist, maxnorm

"typealias fora static vector type; currently `StaticArrays.SVector`"
const SVec = SVector
"typealias fora static matrix type; currently `StaticArrays.SMatrix`"
const SMat = SMatrix

"`JVec{T}` : 3-dimensional immutable vector"
const JVec{T} = SVec{3,T}
const JVecF = JVec{Float64}
const JVecI = JVec{Int}


"`JMat{T}` : 3 × 3 immutable matrix"
const JMat{T} = SMatrix{3,3,T,9}
const JMatF = SMatrix{3,3,Float64,9}
const JMatI = JMat{Int}

#

"""
`vecs(V)` : convert (as reference) a `3 x N` matrix or `3*N` vector representing
N vectors (e.g. positions or forces) in R³ to a vector of `JVec` vectors.

If `V` is obtained from a call to  `mat(X)` where `X::Vector{JVec{T}}` then
`vecs(V) === X`.
"""
vecs(V::AbstractMatrix{T}) where {T} = (@assert size(V,1) == 3;
                                        reinterpret(JVec{T}, vec(V)))
vecs(V::AbstractVector{T}) where {T} = reinterpret(JVec{T}, V)
vecs(M::ReshapedArray{T,2,ReinterpretArray{T,1,JVec{T},Array{JVec{T},1}},Tuple{}}
    ) where {T} = M.parent.parent


"""
`mat(X)`: convert (as reference) a vector of
`SVec` to a 3 x N matrix representing those.

### Usage:
```
X = positions(at)          # returns a Vector{<: JVec}
M = positions(at) |> mat   # returns an AbstractMatrix
X === vecs(M)
```
"""
mat(V::AbstractVector{SVec{N,T}}) where {N,T} =
      reshape( reinterpret(T, V), (N, length(V)) )
# mat(X::Base.ReinterpretArray) = reshape(X.parent, 3, :)




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
function xyz(V::AbstractMatrix)
   @assert size(V, 1) == 3
   return (view(V, 1, :), view(V, 2, :), view(V, 3, :))
end

xyz(V::AbstractVector{JVec{T}}) where {T} = xyz(mat(V))

"""
`maxdist(A, B): ` maximum of distances between two ordered sets of objects,
typically positions. `maximum(norm(a - b) for (a,b) in zip(A,B))`
"""
function maxdist(x::AbstractArray, y::AbstractArray)
   @assert length(x) == length(y)
   return maximum( norm(a-b)  for (a,b) in zip(x,y) )
end

"""
`maxdist(X, y): ` maximum of distance of `y` to all items of `X`;
`maximum( norm(x - y) for x in X)`
"""
maxdist(X::AbstractArray{T}, y::T) where {T} = maximum(norm(x - y) for x in X)
maxdist(x::T, X::AbstractArray{T}) where {T} = maxdist(X, x)


"`maximum(norm(y) for y in x);` typically, x is a vector of forces"
maxnorm(X::AbstractVector) = maximum( norm(x) for x in X )
