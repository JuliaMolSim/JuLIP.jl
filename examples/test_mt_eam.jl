using JuLIP, BenchmarkTools, Base.Threads

# basic EAM potential
data = joinpath(dirname(@__FILE__), "..", "data") * "/"
eam_W = JuLIP.Potentials.FinnisSinclair(data*"W-pair-Wang-2014.plt", data*"W-e-dens-Wang-2014.plt")

@show nthreads()

for L in [5, 10, 20]
   let at = bulk(:W, cubic=true) * L
      @show length(at)
      print("      energy : ")
      @btime energy($eam_W, $at)
      print("      forces : ")
      @btime forces($eam_W, $at)
      print("  energy_map : ")
      @btime energy_map($eam_W, $at)
      print("  forces_map : ")
      @btime forces_map($eam_W, $at)
   end
end
