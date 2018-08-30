using Test
using JuLIP
using JuLIP.Potentials
using JuLIP.Testing
using StaticArrays
using JuLIP.Potentials: evaluate_d, evaluate_dd

println("============================================")
println("  Testing pair potential hessian ")
println("============================================")

at = bulk(:Cu, cubic=true)
pp = lennardjones(r0=rnn(:Cu))

println("test the potential (scalar) itself")
r = 0.9*rnn(:Cu) + 3.0*rand(100)
ddJ = [@DD pp(r) for r in r]
ddJh = [((@D pp(r + 1e-4)) - (@D pp(r - 1e-4))) / 2e-4  for r in r]
@show maximum(abs.(ddJ - ddJh))

println("test `hess`; with two test vectors")
for R in ( [0.0,-3.61,-3.61], [-1.805,-1.805,-3.61] )
   r = norm(R)
   @show r, R
   E0 = pp(r)
   # f0 = grad(pp, r, JVecF(R))
   f0 = evaluate_d(pp, r) * R/r
   size(f0)
   # df0 = hess(pp, r, JVecF(R))
   df0 = evaluate_dd(pp, r) * (R/r) * (R/r)' +
         (evaluate_d(pp, r)/r) * (eye(3) - (R/r) * (R/r)')
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
     @printf("%1.1e | %4.2e \n", h, vecnorm(dfh - df0, Inf))
   end
   @printf("-------|--------- \n")
end

println("full finite-difference test for pairpot `hessian`")
at = at * 2
set_pbc!(at, false)
set_constraint!(at, FixedCell(at))
set_calculator!(at, pp)
fdtest_hessian( x->gradient(at, x), x->hessian(at, x), dofs(at) )


println("============================================")
println("  Testing EAM hessian ")
println("============================================")

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
   r = r1
   R = R1
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
   hV[3*(i-1)+(1:3), 3*(j-1)+(1:3)] = hVs[i,j]
end
matR = mat(R)

for p = 3:9
   h = 0.1^p
   hVh = zeros(hV)
   for n = 1:length(matR)
      matR[n] += h
      r = norm.(R)
      dVh = mat(evaluate_d(eam, r, R))[:]
      hVh[:, n] = (dVh - dV) / h
      matR[n] -= h
   end
   @printf("%1.1e | %4.2e \n", h, vecnorm(hVh - hV, Inf))
end

println("full finite-difference test")
fdtest_hessian( x->gradient(at, x), x->hessian(at, x), dofs(at) )


println("============================================")
println("  Testing Stillinger-Weber hessian ")
println("============================================")

# setup a geometry
at = bulk(:Si, cubic=true) * 2
rattle!(at, 0.02)
set_pbc!(at, false)
set_constraint!(at, FixedCell(at))
sw = StillingerWeber()
set_calculator!(at, sw)

println("full finite-difference test")
@test fdtest( x -> energy(at, x), x -> gradient(at, x), dofs(at) )
@test fdtest_hessian( x->gradient(at, x), x->hessian(at, x), dofs(at) )
