
"""
`EMT`: a re-implementation of the `EMT` calculator (a variant of EAM) os ASE
in Julia, largely for fun and comparison with Python, but also to demonstrate
how to implement a multi-component calculator in JuLIP
"""
struct AnalyticEAM <: MSitePotential
   pair::Vector{WrappedPairPotential}
   rho::Vector{WrappedPairPotential}
   embed::Vector{WrappedAnalyticFunction}
   z2ind::Dict{Int, Int}
end

cutoff(calc::AnalyticEAM) = min(cutoff(calc.pair), cutoff(calc.rho))

AnalyticEAM(N::Integer) =
   AnalyticEAM( Vector{WrappedPairPotential}(undef, N),
                Vector{WrappedPairPotential}(undef, N),
                Vector{WrappedAnalyticFunction}(undef, N),
                Dict{Int, Int}() )

function evaluate!(tmp, calc::AnalyticEAM, 𝐑, 𝐙, z0)
   ρ̄ = 0.0
   Es = 0.0
   i0 = calc.z2ind[z0]
   for (R, Z) in zip(𝐑, 𝐙)
      i = calc.z2ind[Z]
      r = norm(R)
      Es += calc.pair[i](r)
      ρ̄ +=  calc.rho[i](r)
   end
   Es += calc.embed[i0](ρ̄)
   return Es
end

function evaluate_d!(dEs, tmp, calc::EMT, 𝐑, 𝐙, z0)
   ρ̄ = 0.0
   i0 = calc.z2ind[z0]
   for (R, Z) in zip(𝐑, 𝐙)
      i = calc.z2ind[Z]
      r = norm(R)
      ρ̄ += calc.rho[i](r)
   end
   dF = @D calc.embed[i0](ρ̄)
   for (j, (R, Z)) in enumerate(zip(𝐑, 𝐙))
      i = calc.z2ind[Z]
      r = norm(R)
      R̂ = R/r
      dpair = @D calc.pair[i](r)
      drho = @D calc.rho[i](r)
      dEs[j] = (dpair + dF * drho) * R̂
   end
   return dEs
end



"embedding function for the Gupta potential"
mutable struct GuptaEmbed <: SimpleFunction
    xi
end
evaluate(p::GuptaEmbed, r) = p.xi * sqrt(r)
evaluate_d(p::GuptaEmbed, r) = 0.5*p.xi ./ sqrt(r)

"""`GuptaPotential`:
    E_i = A ∑_{j ≠ i} v(r_ij) - ξ ∑_i √ ρ_i
        v(r_ij) = exp[ -p (r_ij/r0 - 1) ]
        ρ_i = ∑_{j ≠ i} exp[ -2q (r_ij / r0 - 1) ]
"""
GuptaPotential(A, xi, p, q, r0, TC::Type, TCargs...)  =
    EAMPotential( TC( SimpleExponential(A, p, r0), TCargs... ),      # V
                  TC( SimpleExponential(1.0, 2*q, r0), TCargs...),   # rho
                  GuptaEmbed( xi ) )                                 # embed

gupta_parameters = Dict(
   #          a       D       α      R₀      <<< Gupta notation
   :Cu => (3.6147, 0.3446, 1.3921, 2.838),
   :Ag => (4.0862, 0.3294, 1.3939, 3.096),
   :Au => (4.0785, 0.4826, 1.6166, 3.004),
   :Ni => (3.5238, 0.4279, 1.3917, 2.793)
)


function GuptaEAM()


end




# # data Johnson & Oh (1989)
#
# johnson_params = Dict(
#   #  species    a (Å)    Ec(eV)   ΩB(eV)   ΩG(eV)   A
#     :Li   =>   (3.5087,  1.65,    1.63,     0.77,   8.52),
#     :Na   =>   (4.2906,  1.13,    1.63,     0.68,   7.16),
#     :K    =>   (5.344,   0.941,   1.58,     0.59,   6.71),
#     :V    =>   (3.0399,  5.30,    13.62,    4.17,   0.78),
#     :Nb   =>   (3.3008,  7.47,    19.08,    4.43,   0.52),
#     :Ta   =>   (3.3026,  8.089,   21.66,    7.91,   1.57),
#     :Cr   =>   (2.8845,  4.10,    11.77,    8.71,   0.71),
#     :Mo   =>   (3.150,   6.810,   25.68,    12.28,  0.78),
#     :W    =>   (3.16475, 8.66,    30.65,    15.84,  1.01),
#     :Fe   =>   (2.86645, 4.29,    12.26,    6.53,   2.48)
# )
#
# function JohnsonEAM(sym::Symbol)
#    if !haskey(johnson_params, sym)
#       error("""The Johnson & Oh EAM model does is not defined for $(sym);
#                only the chemical symbols $(keys(johnson_params)) are
#                available""" )
#    end
#    a, Ec, ΩB, ΩG, A = johnson_params[sym]
#    c1 = - 15 * ΩG / (3*A + 2)
#    K3 = c1 * (3/4*A - 0.5*S)   # (6)
#    K2 = c1 * (15/16*A - 3/4*S - 9/8*A*S + 7/8)
#    K1 = - 15 * ΩG * (7/8 - 3/4*S)
#    K0 =
#    V =
#    f = @analytic r ->
#    F = @analytic ρ -> - (Ec-EUF) * (1 - log((ρ/ρe)^n)) * (ρ/ρe)^n
# end
