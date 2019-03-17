# included from Potentials.jl
# part of the module JuLIP.Potentials

using JuLIP: zerovecs, JVecsF, JVecF, JMatF, neighbourlist
using LinearAlgebra: I
using JuLIP.Chemistry: atomic_number

using NeighbourLists

export ZeroPairPotential, PairSitePotential, ZBLPotential,
         LennardJones, lennardjones,
         Morse, morse, grad, hess

function site_energies(pp::PairPotential, at::AbstractAtoms)
   Es = zeros(length(at))
   for (i,_2,r,_3) in pairs(at, cutoff(pp))
      Es[i] += 0.5 * pp(r)
   end
   return Es
end

import Base.sum

sum(V::PairPotential, r) = sum( V(s) for s in r )

function sum(V::PairPotential, r::Vector{T}) where T <: Real
   E = 0.0
   @simd for n = 1:length(r)
      @inbounds E += V(r[n])
   end
   return E
end

# a simplified way to calculate gradients of pair potentials
@inline grad(V::PairPotential, r::Real, R::JVec) = (evaluate_d(V, r) / r) * R

# function energy(V::PairPotential, at::AbstractAtoms)
#    E = 0.0
#    for (_₁, _₂, r, _₃) in pairs(at, cutoff(V))
#       E += V(r)
#    end
#    return 0.5 * E
# end

function energy(V::PairPotential, at::AbstractAtoms)
   nlist = neighbourlist(at, cutoff(V))::PairList
   return 0.5 * sum(V, nlist.r)
end


# function forces(V::PairPotential, at::AbstractAtoms)
#    dE = zerovecs(length(at))
#    for (i,j,r,R) in pairs(at, cutoff(V))
#       dE[i] += @GRAD V(r, R)
#    end
#    return dE
# end

function forces(V::PairPotential, at::AbstractAtoms)
   nlist = neighbourlist(at, cutoff(V))::PairList
   dE = zerovecs(length(at))
   @simd for n = 1:npairs(nlist)
      @inbounds dE[nlist.i[n]] += grad(V, nlist.r[n], nlist.R[n])
   end
   return dE
end


# TODO: rewrite using generator once bug is fixed (???or maybe decide not to bother???)
function virial(pp::PairPotential, at::AbstractAtoms)
   S = zero(JMatF)
   for (_₁, _₂, r, R) in pairs(at, cutoff(pp))
      S -= 0.5 * grad(pp, r, R) * R'  # (((@D pp(r)) / r) * R) * R'
   end
   return S
end


function hess(V::PairPotential, r, R)
   R̂ = R/r
   P = R̂ * R̂'
   dV = (@D V(r))/r
   return ((@DD V(r)) - dV) * P + dV * one(JMatF)
end

# an FF preconditioner for pair potentials
function precon(V::PairPotential, r, R, innerstab=0.1)
   dV = @D V(r)
   hV = @DD V(r)
   # Id = eye(JMatF)
   S = R/r
   return (1-innerstab) * (abs(hV) * S * S' + abs(dV / r) * (I - S * S')) +
             innerstab  * (abs(hV) + abs(dV / r)) * one(JMatF)
end


hessian_pos(V::PairPotential, at::AbstractAtoms) =
      _precon_or_hessian_pos(V, at, hess)

#
# this assembles a hessian or preconditioner as a block-matrix
#
function _precon_or_hessian_pos(V::PairPotential, at::AbstractAtoms, hfun)
   nlist = neighbourlist(at, cutoff(V))
   I, J, Z = Int[], Int[], JMatF[]
   for C in (I, J, Z); sizehint!(C, 2*npairs(nlist)); end
   for (i, j, r, R) in pairs(nlist)
      h = 0.5 * hfun(V, r, R)
      append!(I, (i,  i,  j, j))
      append!(J, (i,  j,  i, j))
      append!(Z, (h, -h, -h, h))
   end
   hE = sparse(I, J, Z, length(at), length(at))
   return hE
end



"""
`LennardJones(σ, e0):` constructs the 6-12 Lennard-Jones potential [wiki](https://en.wikipedia.org/wiki/Lennard-Jones_potential)

   e0 * 4 * ( (σ/r)¹² - (σ/r)⁶ )

Constructor with kw arguments: `LennardJones(; kwargs...)`

* `e0, σ` : standard LJ parameters, default `e0 = 1.0, σ = 1.0`
* `r0` : equilibrium distance - if `r0` is specified then `σ` is ignored
* `a0` : FCC lattice parameter, if `a0` is specified then `σ` is ignored

(`r0, a0` cannot both be specified at the same time)
"""
LennardJones(σ, e0) = (@analytic r -> e0 * 4.0 * ((σ/r)^(12) - (σ/r)^(6)))


function ljparams(; σ=1.0, e0=1.0, r0 = nothing, a0 = nothing)
   if r0 != nothing && a0 != nothing
      error("`LenndardJones`: cannot specify both `r0` and `a0`")
   end
   if a0 != nothing    # r0 = nn-dist = a0/sqrt(2) in FCC
      r0 = a0 / sqrt(2)
   end
   if r0 != nothing    # standard LJ is minimised at r = 2^(1/6)
      σ =  r0 / 2^(1/6)
   end
   return σ, e0
end

LennardJones(; kwargs...) = LennardJones(ljparams(;kwargs...)...)


"""
`lennardjones(; kwargs...)`

simplified constructor for `LennardJones` (note this is type unstable!)

In addition to the `kwargs` of `LennardJones`, this accepts also

* `rcut` : default `:auto` which gives `rcut = (1.9*σ, 2.7*σ)`. Use
`nothing` or `Inf` to specify no cutoff, or specify a tuple or
array with two elements specifying the lower and upper cut-off radii to be
used with `SplineCutoff`.
"""
function lennardjones(; rcut = :auto, kwargs...)
   σ, e0 = ljparams(; kwargs...)
   if (rcut == nothing || rcut == Inf)
      return LennardJones(σ, e0)
   elseif rcut == :auto
      rcut = (1.9*σ, 2.7*σ)
   end
   return SplineCutoff(rcut[1], rcut[2]) * LennardJones(σ, e0)
end

"""
`Morse(A, e0, r0)` or `Morse(;A=4.0, e0=1.0, r0=1.0)`: constructs a
`PairPotential` for
   e0 ( exp( -2 A (r/r0 - 1) ) - 2 exp( - A (r/r0 - 1) ) )
"""
Morse(A, e0, r0) = @analytic(
   r -> e0 * ( exp(-(2.0*A) * (r/r0 - 1.0)) - 2.0 * exp(-A * (r/r0 - 1.0)) ) )
Morse(;A=4.0, e0=1.0, r0=1.0) = Morse(A, e0, r0)

"""
`morse(A=4.0, e0=1.0, r0=1.0, rcut=(1.9*r0, 2.7*r0))`

simplified constructor for `Morse` (type unstable)
"""
morse(;A=4.0, e0=1.0, r0=1.0, rcut=(1.9*r0, 2.7*r0)) = (
   (rcut == nothing || rcut == Inf)
         ?  Morse(A, e0, r0)
         :  SplineCutoff(rcut[1], rcut[2]) * Morse(A, e0, r0) )


"""
`ZeroPairPotential()`: creates a potential that just returns zero
"""
struct ZeroPairPotential <: PairPotential end

@pot ZeroPairPotential

evaluate(p::ZeroPairPotential, r::Float64) = 0.0
evaluate_d(p::ZeroPairPotential, r::Float64) = 0.0
evaluate_dd(p::ZeroPairPotential, r::Float64) = 0.0
cutoff(p::ZeroPairPotential) = 0.0


# ========================================================
# wrapping a pair potential in a site potential

SitePotential(pp::PairPotential) = PairSitePotential(pp)

struct PairSitePotential{P} <: SitePotential
   pp::P
end

@pot PairSitePotential

cutoff(psp::PairSitePotential) = cutoff(psp.pp)

# TODO: get rid of this!
function _sumpair_(pp, r)
   # cant use a generator here since type is not inferred!
   # Watch out for a bugfix
   s = 0.0
   for t in r
      s += pp(t)
   end
   return s
end

# special implementation of site energy and forces for a plain pair potential
evaluate(psp::PairSitePotential, r, R) = 0.5 * _sumpair_(psp.pp, r)

evaluate_d(psp::PairSitePotential, r, R) =
            [ 0.5 * grad(psp.pp, s, S) for (s, S) in zip(r, R) ]



struct MultiPairPotential{TV <: PairPotential}
   V::Matrix{TV}
   Z2V::Dict{Int, Int}
end


# ------------------------------------------------------------------------

# TODO: write more docs + tests for ZBL

"""
Implementation of the ZBL potential to model close approach.
"""
struct ZBLPotential{TV} <: PairPotential
   Z1::Int
   Z2::Int
   V::TV
end

@pot ZBLPotential

@inline evaluate(V::ZBLPotential, args...) = evaluate(V.V, args...)
@inline evaluate_d(V::ZBLPotential, args...) = evaluate_d(V.V, args...)
@inline grad(V::ZBLPotential, args...) = grad(V.V, args...)
cutoff(::ZBLPotential) = Inf

ZBLPotential(Z1::Integer, Z2::Integer) =
   let Z1=Z1, Z2=Z2
      au = 0.8854 * 0.529 / (Z1^0.23 + Z2^0.23)
      ϵ0 = 0.00552634940621
      C = Z1*Z2/(4*π*ϵ0)
      E1, E2, E3, E4 = 0.1818, 0.5099, 0.2802, 0.02817
      A1, A2, A3, A4 = 3.2/au, 0.9423/au, 0.4028/au, 0.2016/au
      V = @analytic(r -> C * (E1*exp(-A1*r) + E2*exp(-A2*r) +
                              E3*exp(-A4*r) + E4*exp(-A4*r) ) / r)
      ZBLPotential(Z1, Z2, V)
   end


ZBLPotential(Z::Integer) = ZBLPotential(Z, Z)
ZBLPotential(s1::Symbol, s2::Symbol) = ZBLPotential(atomic_number(s1), atomic_number(s2))
ZBLPotential(s::Symbol) = ZBLPotential(s, s)

Dict(V::ZBLPotential) = Dict("__id__" => "JuLIP_ZBLPotential",
                             "Z1" => V.Z1,
                             "Z2" => Z.Z2)
ZBLPotential(D::Dict) = ZBLPotential(D["Z1"], D["Z2"])
Base.convert(::Val{:JuLIP_ZBLPotential}, D::Dict) = ZBLPotential(D)
