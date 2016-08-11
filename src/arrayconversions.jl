

using FixedSizeArrays

import Base.convert

export mat, pts, vecs
export JVec, JVecs, JPt, JPts, JVecsPts
export zerovecs, zeropts
export maxdist, maxnorm
export ee

"`JVec{T}` : 3-dimensional immutable vector"
typealias JVec{T} Vec{3,T}

"`JPt{T}` : 3-dimensional immutable point"
typealias JPt{T} Point{3,T}

"`JVecs{T}` : List of 3-dimensional immutable vectors"
typealias JVecs{T} Vector{JVec{T}}

"`JPts{T}` : List of 3-dimensional immutable points"
typealias JPts{T} Vector{JPt{T}}

typealias JVecsPts{T} Union{JVecs{T},JPts{T}}

const e1 = JPt{Float64}((1.0,0.0,0.0))
const e2 = JPt{Float64}((0.0,1.0,0.0))
const e3 = JPt{Float64}((0.0,0.0,1.0))
const EE = [e1;e2;e3]

"returns the i-th canonical basis vector as JPt{Float64} (i = 1,2,3)"
ee(i::Integer) = EE[i]

"""
`pts(X::Matrix)` : convert (as reference) a 3 x N matrix representing
N Points in R³ to a list (vector) of fixed-size-array points.
"""
pts{T}(X::Matrix{T}) = reinterpret(JPt{T}, X, (size(X,2),))
pts{T}(X::Vector{T}) = reinterpret(JPt{T}, X, (length(X) ÷ 3,))

"""
`vecs(V::Matrix)` : convert (as reference) a 3 x N matrix representing
N vectors (e.g. forces) in R³ to a list (vector) of fixed-size-array vectors.
"""
vecs{T}(V::Matrix{T}) = reinterpret(JVec{T}, V, (size(V,2),))
vecs{T}(V::Vector{T}) = reinterpret(JVec{T}, V, (length(V) ÷ 3,))

"""
`mat(X::JPts)` and `mat(X::JVecs)`: convert (as reference) a list (Vector) of
fixed-size-array points or vecs to a 3 x N matrix representing those.

### Usage:
```
X = positions(at)          # returns a Vector{JPt}
X = positions(at) |> mat   # returns a Matrix
```
"""
mat{T}(X::JPts{T}) = reinterpret(T, X, (3, length(X)))
mat{T}(V::JVecs{T}) = reinterpret(T, V, (3, length(V)))

# rewrite all of this in terms of `convert` (TODO: is this needed?)
convert{T}(::Type{Matrix{T}}, X::JPts{T}) = mat(X)
convert{T}(::Type{Matrix{T}}, V::JVecs{T}) = mat(V)
convert{T}(::Type{JPts{T}}, X::Matrix{T}) = pt(X)
convert{T}(::Type{JVecs{T}}, V::Matrix{T}) = vec(V)

# initialise a vector of vecs or points
# zeros{S}(::Type{S <: JVecs{Float64}}, n::Integer) = zeros(3, n) |> vecs
# zeros{S, T}(::Type{S <: JPts{T}}, n::Integer) = zeros(T, 3, n) |> pts

zerovecs(n::Integer) = zerovecs(Float64, n)
zeropts(n::Integer) = zeropts(Float64, n)
zerovecs(T::Type, n::Integer) = zeros(T, 3, n) |> vecs
zeropts(T::Type, n::Integer) = zeros(T, 3, n) |> pts


"""
maximum of distances between two sets of points, usually positions;
`maximum(a - b for (a,b) in zip(x,y))`
"""
function maxdist{T}(x::JPts{T}, y::JPts{T})
   @assert length(x) == length(y)
   ret = 0.0
   for (a,b) in zip(x,y)
      ret = max(ret, norm(a-b))
   end
   return ret
end

"`maximum(norm(y) for y in x);` typically, x is a vector of forces"
maxnorm{T}(x::JVecs{T}) = maximum( norm.(x) )
