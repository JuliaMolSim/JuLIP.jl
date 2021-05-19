

using JuLIP
using Pkg
# @show pathof(JuLIP)

@info("nthreads = $(Threads.nthreads())")

function runtests(at, usethreads)
   JuLIP.usethreads!(usethreads)
   energy(at); forces(at); virial(at)
   tE = minimum( (@elapsed energy(at)) for _=1:5 )
   tF = minimum( (@elapsed forces(at)) for _=1:5 )
   tV = minimum( (@elapsed virial(at)) for _=1:5 )
   tf = usethreads ? "t" : "f"
   println(tf * " -> E: $tE; F: $tF; V: $tV")
   nothing
end

for mul in [5, 10, 20]
   at = bulk(:Cu, cubic=true) * mul
   set_calculator!(at, JuLIP.Potentials.EMT())
   @info("Run timings with $(length(at)) atoms")
   runtests(at, true)
   runtests(at, false)
end

