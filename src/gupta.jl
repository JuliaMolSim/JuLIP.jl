

"embedding function for the Gupts potential"
type GuptaEmbed <: SimpleFunction
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
   "Cu" => (3.6147, 0.3446, 1.3921, 2.838),
   "Ag" => (4.0862, 0.3294, 1.3939, 3.096),
   "Au" => (4.0785, 0.4826, 1.6166, 3.004),
   "Ni" => (3.5238, 0.4279, 1.3917, 2.793)
)
