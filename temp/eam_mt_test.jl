

using ASE
using JuLIP
using BenchmarkTools
using Main.Threads
using DataDeps
using Profile 
using ProfileView

register(DataDep(
    "JuLIP_testpots",
    "A few EAM potentials for testing",
    "https://www.dropbox.com/s/leub1c9ft1mm9fg/JuLIP_data.zip?dl=1",
    post_fetch_method = file -> run(`unzip $file`)
    ))
test_pots = joinpath(datadep"JuLIP_testpots", "JuLIP_data") * "/"
Ni = test_pots * "Ni.eam.fs"

eam0 = EAM(Ni) 
eam1 = EAM(Ni; s = 1e-3) 
display(eam0.ϕ[1].f.spl)
display(eam1.ϕ[1].f.spl)

eam = eam1 

@show Main.Threads.nthreads()
@show JuLIP.nthreads()



function evaln(eam, at, N)
    E = zeros(Main.Threads.nthreads())
    F = [ forces(eam, at) * 0  for _ = 1:Main.Threads.nthreads() ]
    @threads :static for n = 1:N 
        tid = threadid()
        at1 = deepcopy(at) 
        rattle!(at, 0.1)
        E[tid] += energy(eam, at1) / N 
        F[tid] += forces(eam, at1) / N 
    end 
    return sum(E), sum(F)
end

at = bulk(:Ni, cubic=true) * 7

@info("Energy EAM0")
@btime energy($eam0, $at)

@info("Energy EAM1 (low acc)")
@btime energy($eam1, $at)

@info("Forces")
@btime forces($eam0, $at)
@btime forces($eam1, $at)

##

Profile.clear() 
@profile energy(eam, at)

##

tmp = [JuLIP.alloc_temp(eam, at) for i in 1:JuLIP.nthreads()]
@time JuLIP.energy!(tmp, eam, at)

@code_warntype JuLIP.energy!(tmp, eam, at)


##

nlist = neighbourlist(at, cutoff(eam))
Js, Rs = JuLIP.neigs(nlist, 1)
Zs = at.Z[Js]
z0 = at.Z[1]

JuLIP.evaluate!(tmp[1], eam, Rs, Zs, z0)
@btime JuLIP.evaluate!($(tmp[1]), $eam, $Rs, $Zs, $z0)
@code_warntype JuLIP.evaluate!(tmp[1], eam, Rs, Zs, z0)

##

function runn(N, f!, args...)
    s = 0.0
    for n = 1:N 
        s += 0.0001 * f!(args...)
    end
    return s 
end 

##

Profile.clear()
@profile runn(1000, JuLIP.evaluate!, tmp[1], eam, Rs, Zs, z0)

##

spl = eam.ϕ[1].f.spl
r = rnn(:Ni)
@btime $spl($r)


Profile.clear() 

@profile runn(100_000, spl, r);



##


ProfileView.view()

