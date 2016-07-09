
# # import some faster exponential for fast potential assembly
# try
#     import AppleAccelerate
#     function _fast_exponential(Lc, Rc, r)
#         c2 = Rc-r
#         AppleAccelerate.rec!(c2, c2)
#         c2 = Lc * c2
#         AppleAccelerate.exp!(c2, c2)
#         return c2
#     end
# catch
#     _fast_exponential(Lc, Rc, r) = exp( Lc ./ (Rc - r) )
# end



###########################################################################
## Implementation of some basic calculators using MatSciPy.NeighborList
###########################################################################


"""`simple_binsum` : this is a placeholder for a more general function,
`binsum`, which still needs to be written! Here, it is assumed that
`size(A, 1) = 3`, and  only summation along the second dimension is allowed.
"""
function simple_binsum{TI <: Integer, TF <: AbstractFloat}(i::Vector{TI},
                                                           A::Matrix{TF})
    if size(A, 1) != 3
        error("simple_binsum: need size(A,1) = 3")
    end
    if size(A, 2) != length(i)
        error("simple_binsum: need size(A,2) = length(i)")
    end
    B = zeros(TF, 3, maximum(i))
    for m = 1:size(A,1)
        # @inbounds @simd
        for n = 1:length(i)
            B[m, i[n]] = B[m,i[n]] + A[m, n]
        end
    end
    return B
end

function simple_binsum{TI <: Integer, TF <: AbstractFloat}(i::Vector{TI},
                                                           A::Vector{TF})
    if length(A) != length(i)
        error("simple_binsum: need length(A) = length(i)")
    end
    B = zeros(TF, maximum(i))
    # this ought to be a SIMD loop. but that gives me a wrong answer! why?!
    for n = 1:length(i)
        B[i[n]] += + A[n]
    end
    return B
end



"""`PairCalculator` : basic calculator for pair potentials.
"""
type PairCalculator <: AbstractCalculator
    pp::PairPotential
end

import Potentials.cutoff
cutoff(calc::PairCalculator) = cutoff(calc.pp)

function potential_energy(at::ASEAtoms, calc::PairCalculator)
    r = neighbour_list(at, "d", cutoff(calc))
    return sum( calc.pp(r) )
end

function potential_energy_d(at::ASEAtoms, calc::PairCalculator)
    i, r, R = neighbour_list(at, "idD", cutoff(calc))
    return - 2.0 * simple_binsum(i, @GRAD calc.pp(r, R') )
end

forces(at::ASEAtoms, calc::PairCalculator) = - potential_energy_d(at, calc)


"`EAMCalculator` : basic calculator using the `EAMPotential` type"
type EAMCalculator <: AbstractCalculator
    p::EAMPotential
end

cutoff(calc::EAMCalculator) = max(cutoff(calc.p.V), cutoff(calc.p.rho))

function potential_energy(at::ASEAtoms, calc::EAMCalculator)
    i, r = neighbour_list(at, "id", cutoff(calc))
    return ( sum(calc.p.V(r))
             + sum( calc.p.embed( simple_binsum( i, calc.p.rho(r) ) ) ) )
end

function potential_energy_d(at::ASEAtoms, calc::EAMCalculator)
    i, j, r, R = neighbour_list(at, "ijdD", cutoff(calc))
    # pair potential component
    G = - 2.0 * simple_binsum(i, @GRAD calc.p.V(r, R'))
    # EAM component
    dF = @D calc.p.embed( simple_binsum(i, calc.p.rho(r)) )
    dF_drho = dF[i]' .* (@GRAD calc.p.rho(r, R'))
    G += simple_binsum(j, dF_drho) - simple_binsum(i, dF_drho)
    return G
end




# ###########################################################################
# ## Some Calculators Optimised for use with MatSciPy.NeighbourList
# ###########################################################################


# """
# `lennardjones_old(at::ASEAtoms, nlist::NeighbourList;
#                       r0=1.0, e0=1.0, quantities="EG")`

# A fast LJ assembly which exploits the `@simd` and `@inbounds` macros as well
# as the specific structure of `MatSciPy.NeighbourList`.

# **This has not yet been tested for correctness!**

# **This version does not have a cutoff radius!**
# """
# function lennardjones_old(at::ASEAtoms, nlist::NeighbourList;
#                       r0=1.0, e0=1.0, quantities="EG")
#     update!(nlist, at)
#     E = 0.0
#     r = nlist.Q['d']::Vector{Float64}
#     R = nlist.Q['D']::Array{Float64,2}
#     i = nlist.Q['i']::Vector{Int32}
#     t = Vector{Float64}(length(r))

#     @simd for n = 1:length(t) @inbounds begin
#         t[n] = r0 / r[n]
#         t[n] = t[n]*t[n]*t[n]
#         t[n] = t[n]*t[n]
#         E = E + t[n]*(t[n]-2.0)
#     end end
#     E *= e0

#     if 'G' in quantities
#         G = zeros(3, i[end])
#         @simd for n = 1:length(t) @inbounds begin
#             t[n] = e0 * 24.0 * t[n]*(t[n]-1.0) / (r[n]*r[n])
#             G[1,i[n]] = G[1,i[n]] + t[n] * R[n,1]
#             G[2,i[n]] = G[2,i[n]] + t[n] * R[n,2]
#             G[3,i[n]] = G[3,i[n]] + t[n] * R[n,3]
#         end end
#     end

#     ret = ()
#     for c in quantities
#         if c == 'E'
#             ret = tuple(ret..., E)
#         elseif c == 'G'
#             ret = tuple(ret..., G)
#         end
#     end
#     return ret
# end




# """
# `lennardjones(at::ASEAtoms, nlist::NeighbourList;
#                       r0=1.0, e0=1.0, Rc=2.7, Lc=1.0, quantities="EG")`

# A fast LJ assembly which exploits the specific structure of the
#  `MatSciPy.NeighbourList` to use  `@simd`, `@inbounds`, `@fastmath` macros
# and the `AppleAccalerate` package.

# **This has not yet been tested for correctness!**
# """
# function lennardjones(at::ASEAtoms, nlist::NeighbourList;
#                       r0=1.0, e0=1.0, Rc=2.7, Lc = 1.0,  quantities="EG")
#     update!(nlist, at)
#     E = 0.0
#     r = nlist.Q['d']::Vector{Float64}
#     R = nlist.Q['D']::Array{Float64,2}
#     i = nlist.Q['i']::Vector{Int32}
#     t = Vector{Float64}(length(r))
#     c1 = Vector{Float64}(length(r))
#     # c2 = Vector{Float64}(length(r))
#     Rc_ = Rc-0.001

#     # c2 = exp(Lc ./ (Rc-r))
#     c2 = _fast_exponential(Lc, Rc, r)

#     @fastmath @inbounds @simd for n = 1:length(t)
#         # c2[n] = exp( Lc / (Rc-r[n]) )
#         c1[n] = (1.0 / (1.0 + c2[n]))  * (r[n] >= Rc_)
#         t[n] = r0 / r[n]
#         t[n] = t[n]*t[n]*t[n]
#         t[n] = t[n]*t[n]
#         E = E + t[n] * (t[n]-2.0) * c1[n]
#     end
#     E *= e0

#     if 'G' in quantities
#         G = zeros(3, i[end])
#         @fastmath @inbounds @simd for n = 1:length(t)
#             c2[n] = -Lc * c2[n] * c1[n] / ( (Rc-r[n])*(Rc-r[n]) )
#             c2[n] = c2[n] * t[n] * (t[n]-2.0)
#             t[n] = e0 * 24.0 * t[n] * (t[n]-1.0) / (r[n]*r[n])
#             t[n] = t[n] * c1[n] + c2[n]
#             G[1,i[n]] = G[1,i[n]] + t[n] * R[n,1]
#             G[2,i[n]] = G[2,i[n]] + t[n] * R[n,2]
#             G[3,i[n]] = G[3,i[n]] + t[n] * R[n,3]
#         end
#     end

#     ret = ()
#     for c in quantities
#         if c == 'E'
#             ret = tuple(ret..., E)
#         elseif c == 'G'
#             ret = tuple(ret..., G)
#         end
#     end
#     return ret
# end
