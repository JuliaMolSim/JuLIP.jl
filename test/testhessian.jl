using JuLIP
using JuLIP.Potentials
using JuLIP.Testing
using JuLIP.ASE
using StaticArrays

println("============================================")
println("  Testing pair potential hessian ")
println("============================================")

at = bulk("Cu", cubic=true)
pp = lennardjones(r0=rnn("Cu"))
nlist = neighbourlist(at, cutoff(pp))
for (i,j,r,R,_) in bonds(nlist)
  err1 = Float64[]
  err2 = Float64[]
  E0 = pp(r)
  f0 = grad(pp, r, R)
  size(f0)
  df0 = hess(pp, r, R)
  Rn = [0.0 0.0 0.0];
  for i = 1:3
    Rn[i] = R[i];
  end
  dfh = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
  fh = [0.0, 0.0, 0.0]
  for p = 2:11
    h = 0.1^p
    for n = 1:3
      Rn[n] += h
      rn = vecnorm(Rn)
      fh[n] = (pp(rn)-E0)/h
      dfh[:, n] = (grad(pp, rn, @SVector [Rn[1], Rn[2], Rn[3]])-f0)/h
      Rn[n] -= h
    end
    push!(err1, vecnorm(fh-f0, Inf))
    push!(err2, vecnorm(dfh' - df0, Inf))
    @printf("%1.1e | %4.2e  %4.2e \n", h, err1[end], err2[end])
  end
  @printf("-------|--------- \n")
  @show r
  @show R
  @show f0
  @show df0
  sleep(3)
end

set_pbc!(at, false)set_constraint!(at, FixedCell(at))
set_calculator!(at, lennardjones(r0=rnn("Cu")))
fdtest_hessian( x->gradient(at, x), x->hessian(at, x), dofs(at) )
