using JuLIP
using JuLIP.Potentials
using JuLIP.Testing
using JuLIP.ASE

pp = LennardJones(1.0,1.0)

println("============================================")
println("  Testing pair potential implementations ")
println("============================================")

r = linspace(0.8, 4.0, 100)

function fdtest_hessian(pp, r, verbose=verbose)
function fdtest_hessian(F::Function, dF::Function, x; verbose=true)
     errors = Float64[]
     E = F(x)
     dE = dF(x)
     # loop through finite-difference step-lengths
     @printf("---------|----------- \n")
     @printf("    h    | error \n")
     @printf("---------|----------- \n")
     for p = 2:11
        h = 0.1^p
        dEh = copy(dE)
        for n = 1:length(dE)
           x[n] += h
           dEh[n] = (F(x) - E) / h
           x[n] -= h
        end
        push!(errors, vecnorm(dE - dEh, Inf))
        @printf(" %1.1e | %4.2e  \n", h, errors[end])
     end
     @printf("---------|----------- \n")
     if minimum(errors) <= 1e-3 * maximum(errors)
        println("passed")
        return true
     else
        warn("""It seems the finite-difference test has failed, which indicates
        that there is an inconsistency between the function and gradient
        evaluation. Please double-check this manually / visually. (It is
        also possible that the function being tested is poorly scaled.)""")
        return false
     end
  end

println("--------------------------------")
println(pp)
println("--------------------------------")
fdtest_hessian(pp, r, verbose=verbose)
