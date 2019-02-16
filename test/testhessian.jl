using Test
using JuLIP
using JuLIP.Potentials
using JuLIP.Testing
using StaticArrays
using JuLIP.Potentials: evaluate_d, evaluate_dd
using LinearAlgebra

h2("Testing pair potential hessian")

at = bulk(:Cu, cubic=true)
pp = lennardjones(r0=rnn(:Cu))

h3("test the potential (scalar) itself")
rr = 0.9*rnn(:Cu) .+ 3.0*rand(100)
ddJ = [@DD pp(r) for r in rr]
ddJh = [((@D pp(r + 1e-4)) - (@D pp(r - 1e-4))) / 2e-4  for r in rr]
println( @test (@show maximum(abs.(ddJ .- ddJh))) < 1e-4 )

h3("test `hess`; with two test vectors")
for R in ( [0.0,-3.61,-3.61], [-1.805,-1.805,-3.61] )
   r = norm(R)
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

h3("full finite-difference test for pairpot `hessian`")
at = at * 2
set_pbc!(at, false)
set_constraint!(at, FixedCell(at))
set_calculator!(at, pp)
println(@test fdtest_hessian( x->JuLIP.gradient(at, x), x->hessian(at, x), dofs(at) ))


h2("Testing EAM hessian")
# setup a geometry
at = bulk(:Fe, cubic=true) * 2
set_pbc!(at, false)
set_constraint!(at, FixedCell(at))
dir = joinpath(dirname(@__FILE__), "..", "data") * "/"
eam = eam_Fe
set_calculator!(at, eam)

println("test a single stencil")
r = []
R = []
for (_1, _2, r1, R1) in sites(at, cutoff(eam))
   global r = r1
   global R = R1
   break
end

r = r[1:3]
R = R[1:3]

# evaluate site gradient and hessian
dVs = evaluate_d(eam, r, R)
hVs = hess(eam, r, R)
# and convert them to vector form
dV = mat(dVs)[:]
hV = zeros(3*size(hVs,1), 3*size(hVs,2))
for i = 1:size(hVs,1), j = 1:size(hVs,2)
   hV[3*(i-1).+(1:3), 3*(j-1).+(1:3)] = hVs[i,j]
end
matR = mat(R)

for p = 3:9
   h = 0.1^p
   hVh = fill(0.0, size(hV))
   for n = 1:length(matR)
      matR[n] += h
      r = norm.(R)
      dVh = mat(evaluate_d(eam, r, R))[:]
      hVh[:, n] = (dVh - dV) / h
      matR[n] -= h
   end
   @printf("%1.1e | %4.2e \n", h, norm(hVh - hV, Inf))
end

h3("full finite-difference test")
println(@test fdtest( x -> energy(at, x), x -> JuLIP.gradient(at, x), dofs(at) ))
@warn "fdtest_hessian test has been turned off!"  # TODO: put back in
# println(@test fdtest_hessian( x->gradient(at, x), x->hessian(at, x), dofs(at) ))


h2("Testing Stillinger-Weber hessian")

# setup a geometry
at = bulk(:Si, cubic=true) * 2
rattle!(at, 0.02)
set_pbc!(at, false)
set_constraint!(at, FixedCell(at))
sw = StillingerWeber()
set_calculator!(at, sw)

h3("full finite-difference test")
println(@test fdtest( x -> energy(at, x), x -> JuLIP.gradient(at, x), dofs(at) ))
println(@test fdtest_hessian( x->JuLIP.gradient(at, x), x->hessian(at, x), dofs(at) ))
