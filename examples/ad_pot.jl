using JuLIP, BenchmarkTools, ReverseDiff
using JuLIP.Potentials: evaluate_d!

const r0 = rnn(:Cu)
const rcut = 2.5 * r0

# --------------- EAM Potential based on ForwardDiff -------------------


eam1 = let r0 = r0, rcut = rcut
   ρ(r) = exp(-3*(r/r0-1.0)) - exp(-3*(rcut/r0-1.0) ) +
                  3 * exp( - 3 * (rcut/r0 - 1.0) ) * (r/r0 - rcut/r0)
   Rs -> sqrt(sum(ρ ∘ norm, R))
end

V_fd = ADPotential(eam1, rcut)

# --------------- EAM Potential based on ReverseDiff -------------------
# Still need to figure out how to incorporate ReverseDiff into
# JuLIP, so there are no convenience wrappers for this

eam_mat = let r0 = r0, rcut = rcut
   ρ(r) = exp(-3*(r/r0-1.0)) - exp(-3*(rcut/r0-1.0) ) +
                  3 * exp( - 3 * (rcut/r0 - 1.0) ) * (r/r0 - rcut/r0)
   Rs -> sqrt(sum(ρ(norm(@view Rs[:, n])) for n = 1:size(Rs, 2)))
end

eam_rd = Rs -> eam_mat(mat(Rs))
eam_rd_d = Rs -> ReverseDiff.gradient(
      s -> eam_mat(reshape(s, (3, length(s) ÷ 3))), mat(Rs)[:]) |> vecs

# --------------- EAM Potential based on JuLIP Machinery -------------------
# hand-coded EAM, only ρ, ϕ, √ are differentiated symbolically
# (i.e. source code transformation, and NOT ForwardDiff / ReverseDiff)
V_j = EAM( ZeroPairPotential(),
           (@analytic r -> exp(-3*(r/r0-1.0))) * C1Shift(rcut),
           (@analytic t -> sqrt(t)) )

R = []
F = []
println("Timing of Site-Potential Calls")
for N in [10, 20, 40, 80]
   println("$N neighbours")
   R = r0 + rand(JVecF, N)
   F = similar(R)
   r = norm.(R)
   @assert V_fd(R) ≈ V_j(r, R)
   print("  ForwardDiff:   V: ")
   @btime V_fd($R)
   print("  ReverseDiff:   V: ")
   @btime eam_rd($R)
   print("        JuLIP:   V: ")
   @btime V_j($R)
   print("  ForwardDiff:  ∇V: ")
   @btime (@D V_fd($R))
   print("  ReverseDiff:  ∇V: ")
   @btime eam_rd_d($R)
   print("        JuLIP:  ∇V: ")
   @btime evaluate_d!($F, $V_j, $r, $R)
end
