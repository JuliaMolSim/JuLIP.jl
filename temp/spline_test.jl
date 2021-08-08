


using Dierckx 
using ASE 
using JuLIP
using BenchmarkTools
using Main.Threads
using DataDeps
using Profile 
using ProfileView
using Interpolations
using LinearAlgebra
using Test

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
eam = eam1 
display(eam0.ϕ[1].f.spl)
display(eam1.ϕ[1].f.spl)

@show Main.Threads.nthreads()
@show JuLIP.nthreads()

##



spl_dx = eam0.ϕ[1].f.spl
knots = spl_dx.t[5:end-4] 
@assert norm(diff(diff(knots))) < 1e-12 
h = knots[2]-knots[1]
knots = (knots[1]-h):h:(knots[end]+3*h/2)
vals = spl_dx.(knots)
spl_i = interpolate(vals, BSpline(Cubic(Line(OnGrid()))))
spl_i = scale(spl_i, knots)

##

r = 2.5 + rand() - 0.5 
spl_dx(r)
spl_i(r)

@info("Dierckx: eval, grad")
@btime ($spl_dx)($r)
@btime Dierckx.derivative($spl_dx, $r)

@info("Interpolations, eval, grad")
@btime ($spl_i)($r)
@btime Interpolations.gradient($spl_i, $r)[1]

##

function test(r)
    val_dx = spl_dx(r) 
    val_i = spl_i(r) 
    err = minimum([ abs(val_dx - val_i), 
                    abs(val_dx - val_i) / (abs(val_i) + eps()) ]
                  )
    JuLIP.Testing.print_tf(@test err < 1e-12)
end


@info("test for correctness")
@info("  [1] random evaluation")
for _ = 1:100 
    r = knots[1] + rand() * (knots[end]-knots[1])
    test(r)
end


@info("test for correctness")
@info("  [2] close to cutoff")
for _ = 1:100 
    r = knots[end] - rand() * 1e-4
    test(r)
end

@info("test for correctness")
@info("  [3] even closer to cutoff")
for _ = 1:100 
    r = knots[end] - rand() * 1e-12
    test(r)
end


