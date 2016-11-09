using JuLIP
using JuLIP.Potentials
using JuLIP.Testing
using JuLIP.ASE


println("============================================")
println("  Testing pair potential hessian ")
println("============================================")

at = bulk("Cu", cubic=true) * 3
set_pbc!(at, false)
set_constraint!(at, FixedCell(at))
set_calculator!(at, lennardjones(r0=rnn("Cu")))
fdtest_hessian( x->gradient(at, x), x->hessian(at, x), dofs(at) )
