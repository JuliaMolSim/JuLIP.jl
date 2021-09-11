using Test, JuLIP, StaticArrays, Printf, LinearAlgebra
using JuLIP.Potentials
using JuLIP.Testing
using JuLIP.Potentials: evaluate_d, evaluate_dd

##

h2("Testing pair potential hessian")

at = bulk(:Cu, cubic=true)
pp = lennardjones(r0=rnn(:Cu))

h3("test the potential (scalar) itself")
rr = 0.9*rnn(:Cu) .+ 3.0*rand(100)
ddJ = [@DD pp(r) for r in rr]
ddJh = [((@D pp(r + 1e-4)) - (@D pp(r - 1e-4))) / 2e-4  for r in rr]
println( @test (@show maximum(abs.(ddJ .- ddJh))) < 1e-4 )

##

h3("test `hess`; with two test vectors")
for R in ( [0.0,-3.61,-3.61], [-1.805,-1.805,-3.61] )
   local r = norm(R)
   @show r, R
   E0 = pp(r)
   # f0 = grad(pp, r, JVecF(R))
   f0 = evaluate_d(pp, r) * R/r
   size(f0)
   # df0 = hess(pp, r, JVecF(R))
   df0 = evaluate_dd(pp, r) * (R/r) * (R/r)' +
         (evaluate_d(pp, r)/r) * (I - (R/r) * (R/r)')
   dfh = zeros(3,3)
   fh = zeros(3)
   for p = 2:11
     h = 0.1^p
     for n = 1:3
       R[n] += h
      #  fh[n] = (evaluate(pp, norm(R)) - E0)
       dfh[:, n] = (evaluate_d(pp, norm(R)) * R/norm(R) - f0) / h
         # (grad(pp, r, JVecF(R)) - f0) / h
       R[n] -= h
     end
     @printf("%1.1e | %4.2e \n", h, norm(dfh - df0, Inf))
   end
   @printf("-------|--------- \n")
end

##


h3("full finite-difference test for pairpot `hessian`")
h3("without PBC")
at = at * 2
set_pbc!(at, false)
set_calculator!(at, pp)
println(@test fdtest_hessian( x->JuLIP.gradient(at, x), x->hessian(at, x), dofs(at) ))

h3("full finite-difference test for pairpot `hessian`")
h3("with PBC")
set_pbc!(at, true)
println(@test fdtest_hessian( x->JuLIP.gradient(at, x), x->hessian(at, x), dofs(at) ))

##

h2("Testing EAM hessian")
# setup a geometry
at = bulk(:Fe, cubic=true) * 2
set_pbc!(at, false)
eam = eam_Fe
set_calculator!(at, eam)
rattle!(at, 0.1)   # tests pass with 0.0!!

h3("test a single stencil")
r = []
R = []
for (idx, _j, R1) in sites(at, cutoff(eam))
   if idx == 3
      global r = norm(R1)
      global R = R1
      break
   end
end

# evaluate site gradient and hessian
dVs = evaluate_d(eam, R, at.Z, Potentials.i2z(eam, 1))
hVs = evaluate_dd(eam, R, at.Z, Potentials.i2z(eam, 1))
# and convert them to vector form
dV = mat(dVs)[:]
hV = zeros(3*size(hVs,1), 3*size(hVs,2))
for i = 1:size(hVs,1), j = 1:size(hVs,2)
   hV[3*(i-1).+(1:3), 3*(j-1).+(1:3)] = hVs[i,j]
end
matR = mat(R)

errs = []
for p = 2:9
   h = 0.1^p
   hVh = fill(0.0, size(hV))
   for n = 1:length(matR)
      matR[n] += h
      local r = norm.(R)
      dVh = mat(evaluate_d(eam, R, at.Z, Potentials.i2z(eam, 1)))[:]
      hVh[:, n] = (dVh - dV) / h
      matR[n] -= h
   end
   push!(errs, norm(hVh - hV, Inf))
   @printf("%1.1e | %4.2e \n", h, errs[end])
end
println(@test /(extrema(errs)...) < 1e-3)

##

h3("full finite-difference test ...")

h3(" ... EAM forces")
println(@test fdtest( x -> energy(at, x), x -> JuLIP.gradient(at, x), dofs(at) ))

h3(" ... EAM hessian")
println(@test fdtest_hessian( x->gradient(at, x), x->hessian(at, x), dofs(at) ))

h3(" ... EAM hessian with PBC")
set_pbc!(at, true)
println(@test fdtest_hessian( x->gradient(at, x), x->hessian(at, x), dofs(at) ))

##

h2("Testing Stillinger-Weber hessian")

# setup a geometry
at = bulk(:Si, cubic=true) * 2
rattle!(at, 0.02)
set_pbc!(at, false)
sw = StillingerWeber()
set_calculator!(at, sw)

h3("full finite-difference test ...")
h3(" ... SW forces")
println(@test fdtest( x -> energy(at, x), x -> JuLIP.gradient(at, x), dofs(at) ))

h3(" ... SW hessian")
println(@test fdtest_hessian( x->JuLIP.gradient(at, x), x->hessian(at, x), dofs(at) ))

h3(" ... SW hessian with PBC")
set_pbc!(at, true)
println(@test fdtest_hessian( x->JuLIP.gradient(at, x), x->hessian(at, x), dofs(at) ))
