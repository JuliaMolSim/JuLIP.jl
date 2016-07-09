
import Base.convert

using FixedSizeArrays

export mat, pts, vecs
export JVec, JVecs, JPt, JPts
# export @mat

"`JVec{T}` : 3-dimensional immutable vector"
typealias JVec{T} Vec{3,T}

"`JPt{T}` : 3-dimensional immutable point"
typealias JPt{T} Point{3,T}

"`JVecs{T}` : List of 3-dimensional immutable vectors"
typealias JVecs{T} Vector{JVec{T}}

"`JPts{T}` : List of 3-dimensional immutable points"
typealias JPts{T} Vector{JPt{T}}

"""
`mat2pt(X::Matrix)` : convert (as reference) a 3 x N matrix representing
N Points in R³ to a list (vector) of fixed-size-array points.
"""
pts{T}(X::Matrix{T}) = reinterpret(JPt{T}, X, (size(X,2),))

"""
`mat2vec(V::Matrix)` : convert (as reference) a 3 x N matrix representing
N vectors (e.g. forces) in R³ to a list (vector) of fixed-size-array vectors.
"""
vecs{T}(V::Matrix{T}) = reinterpret(JVec{T}, V, (size(V,2),))

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

# rewrite all of this as automatic conversions
convert{T}(::Type{Matrix{T}}, X::JPts{T}) = mat(X)
convert{T}(::Type{Matrix{T}}, V::JVecs{T}) = mat(V)
convert{T}(::Type{JPts{T}}, X::Matrix{T}) = pt(X)
convert{T}(::Type{JVecs{T}}, V::Matrix{T}) = vec(V)


# finally create a macro to avoid having to wrap calls
# into pt2mat vec2mat etc.
"""
`@mat`: macro to conveniently convert a list of fixed-size points or vec
to a matrix. At the moment this only works with `T=Float64` for other
floating point types, use `convert`, `pt2mat` or `vec2mat`.

### Usage:
```
at = Atoms()  # generate an atoms object
positions(at) # return a Vector of JPt (immutable)
@mat positions(at)   # return same as a 3 x N  Matrix.
```
"""
macro mat(fsig)
   return Expr(:call, :convert, Matrix{Float64}, fsig)
end
