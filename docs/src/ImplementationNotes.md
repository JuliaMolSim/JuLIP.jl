
# Implementation Notes

TODO: these are very out of date and need an update

## Prototypes

For commonly used functions such as `set_positions!` etc, a prototype is
defined which just throws an error. The point of this prototype is that
(a) it can be imported from different packages / sub-packages, and (b)
it provides documentation. This is done using `@protofun`.

## Storage of positions, forces, etc

Atomic positions are stored either as a `Vector{Point{DIM,T}}` or as a
`Matrix{T}` of dimension `DIM x Npoints`, where normally `DIM == 3` and
`T == Float64`. The conversion between these can be done for free using
`reinterpret`.

Note that `ASE` internally stores positions as a `Npoints x DIM` matrix, but
this memory layout is not efficient for codes that can exploit fast loops.

The vector of forces (or negative forces aka gradient) is stored either
as a `Vector{Vec{DIM,T}}` or as a  `Matrix{T}` or dimensions `DIM x Npoints`.

The types `Point, Vec` are from the `FixedSizeArrays` package and should in
principle allow fast linear algebra operations on small arrays without
having to write explicit loops.

## Neighbour lists

Because the `matscipy` neighbour list is so fast we don't bother ever storing
and updating the neighbourlist. Instead a calculator can just
