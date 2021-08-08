

using ASE
using JuLIP
using BenchmarkTools
using Main.Threads
using DataDeps

register(DataDep(
    "JuLIP_testpots",
    "A few EAM potentials for testing",
    "https://www.dropbox.com/s/leub1c9ft1mm9fg/JuLIP_data.zip?dl=1",
    post_fetch_method = file -> run(`unzip $file`)
    ))
test_pots = joinpath(datadep"JuLIP_testpots", "JuLIP_data") * "/"
Ni = test_pots * "Ni.eam.fs"
eam = EAM(Ni) 

@show Main.Threads.nthreads()
@show JuLIP.nthreads()



function evaln(eam, at, N)
    E = zeros(Main.Threads.nthreads())
    F = [ forces(eam, at) * 0  for _ = 1:Main.Threads.nthreads() ]
    @threads for n = 1:N 
        tid = threadid()
        at1 = deepcopy(at) 
        rattle!(at, 0.1)
        E[tid] += energy(eam, at1) / N 
        F[tid] += forces(eam, at1) / N 
    end 
    return sum(E), sum(F)
end

at = bulk(:Ni, cubic=true) * 7
@btime energy($eam, $at)
@btime forces($eam, $at)

# evaln(eam, at, 3)
