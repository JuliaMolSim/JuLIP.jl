
# testing performance of AD
# conclusion is that it scales linearly with dimension;
# however it is a factor 100 slower than the objective.

using BenchmarkTools
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile
import ForwardDiff

ϕ(r) = exp( - 6 * (abs(r)-1)) - 2 * exp( - 3 * (abs(r)-1))
dϕ(r) = ForwardDiff.derivative(ϕ, r)
ρ(r) = exp( - 4 * (abs(r)-1) )
dρ(r) = ForwardDiff.derivative(ρ, r)
ψ(t) = sqrt(t)
dψ(t) = ForwardDiff.derivative(ψ, t)

function get_rho(x)
   N = length(x)
   rho = zeros(eltype(x), N)
   for n = 2:N
      t = ρ(x[n]-x[n-1])
      rho[n-1] += t
      rho[n] += 1
   end
   for n = 3:N
      t = ρ(x[n] - x[n-2])
      rho[n] += t
      rho[n-2] += t
   end
   return rho
end

function F(x)
   N = length(x)
   E = 0.0
   for j = 1:2, n = (1+j):N
      E += ϕ(x[n]-x[n-j])
   end
   return E + sum(ψ, get_rho(x))
end

function dF(x)
   N = length(x)
   dE = zeros(N)
   F = ψ.(get_rho(x))
   for j = 1:2, n = (1+j):N
      dϕ_ = dϕ(x[n]-x[n-j])
      dρ_ = dρ(x[n]-x[n-j])
      a = dϕ_ + (F[n] - F[n-j]) * dρ_
      dE[n] += a
      dE[n-j] -= a
   end
   return dE
end

x = gresult = compiled_f_tape = nothing
for N in (100, 400, 1600)
   # pre-record a GradientTape for `F` using inputs with Float64 elements
   f_tape = GradientTape(F, rand(N))
   # compile `f_tape` into a more optimized representation
   compiled_f_tape = compile(f_tape)
   # some inputs and work buffers to play around with
   gresult = zeros(N)
   x = collect(1:N) + rand(N)
   println("N = $N")
   print("                 Time for F: ")
   @btime F($x)
   print("  Time for DF (ForwardDiff): ")
   @btime ForwardDiff.gradient($F, $x)
   print("  Time for DF (ReverseDiff): ")
   @btime gradient!($gresult, $compiled_f_tape, $x)
   print("        Time for DF(manual): ")
   @btime dF($x)
end

# RESULTS 03/04/2018, MacBook Pro, Julia v0.6.2,
# - ForwardDiff                   0.7.4
# - ReverseDiff                   0.2.0
#
# N = 8000
#                Time for F:
#   564.137 μs (2 allocations: 62.58 KiB)
#   Time for DF (ForwardDiff):
# N = 100
#                  Time for F:   6.191 μs (1 allocation: 896 bytes)
#   Time for DF (ForwardDiff):   409.391 μs (6043 allocations: 659.25 KiB)
#   Time for DF (ReverseDiff):   451.058 μs (0 allocations: 0 bytes)
#         Time for DF(manual):   17.426 μs (3 allocations: 2.63 KiB)
# N = 400
#                  Time for F:   26.206 μs (1 allocation: 3.25 KiB)
#   Time for DF (ForwardDiff):   6.538 ms (96204 allocations: 10.17 MiB)
#   Time for DF (ReverseDiff):   1.999 ms (0 allocations: 0 bytes)
#         Time for DF(manual):   63.400 μs (3 allocations: 9.75 KiB)
# N = 1600
#                  Time for F:   109.590 μs (1 allocation: 12.63 KiB)
#   Time for DF (ForwardDiff):   155.275 ms (1536804 allocations: 162.25 MiB)
#   Time for DF (ReverseDiff):   15.896 ms (0 allocations: 0 bytes)
#         Time for DF(manual):   260.063 μs (3 allocations: 37.88 KiB)
