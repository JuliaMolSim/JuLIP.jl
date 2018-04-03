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

# Timing of Site-Potential Calls
# 10 neighbours
#   ForwardDiff:   V:   1.262 μs (44 allocations: 1.16 KiB)
#   ReverseDiff:   V:   1.123 μs (45 allocations: 1.11 KiB)
#         JuLIP:   V:   197.062 ns (2 allocations: 176 bytes)
#   ForwardDiff:  ∇V:   15.068 μs (147 allocations: 7.89 KiB)
#   ReverseDiff:  ∇V:   346.711 μs (1593 allocations: 49.69 KiB)
#         JuLIP:  ∇V:   324.797 ns (4 allocations: 112 bytes)
# 20 neighbours
#   ForwardDiff:   V:   2.126 μs (84 allocations: 2.17 KiB)
#   ReverseDiff:   V:   2.052 μs (85 allocations: 2.05 KiB)
#         JuLIP:   V:   319.104 ns (2 allocations: 256 bytes)
#   ForwardDiff:  ∇V:   25.205 μs (522 allocations: 19.86 KiB)
#   ReverseDiff:  ∇V:   727.735 μs (3164 allocations: 98.28 KiB)
#         JuLIP:  ∇V:   574.875 ns (4 allocations: 112 bytes)
# 40 neighbours
#   ForwardDiff:   V:   3.914 μs (164 allocations: 4.25 KiB)
#   ReverseDiff:   V:   3.900 μs (165 allocations: 3.92 KiB)
#         JuLIP:   V:   584.110 ns (2 allocations: 464 bytes)
#   ForwardDiff:  ∇V:   61.363 μs (1992 allocations: 60.78 KiB)
#   ReverseDiff:  ∇V:   1.429 ms (6305 allocations: 195.53 KiB)
#         JuLIP:  ∇V:   1.062 μs (4 allocations: 112 bytes)
# 80 neighbours
#   ForwardDiff:   V:   7.824 μs (324 allocations: 8.28 KiB)
#   ReverseDiff:   V:   7.654 μs (325 allocations: 7.67 KiB)
#         JuLIP:   V:   1.093 μs (2 allocations: 752 bytes)
#   ForwardDiff:  ∇V:   192.137 μs (7813 allocations: 209.86 KiB)
#   ReverseDiff:  ∇V:   2.900 ms (12586 allocations: 389.66 KiB)
#         JuLIP:  ∇V:   2.049 μs (4 allocations: 112 bytes)
