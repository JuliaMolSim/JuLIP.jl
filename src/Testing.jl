

"""
This module supplies some functions for testing of the implementation. The
scripts driving these functions area in `LujiaLt/tests`
"""
module Testing

import JuLIP: energy, grad
import JuLIP.Potentials: PairPotential, evaluate, evaluate_d


"generic finite-difference test for scalar F"
function fdtest(F::Function, dF::Function, x)
    errors = Float64[]
    E = F(x)
    dE = dF(x)
    # loop through finite-difference step-lengths
    @printf("---------|----------- \n")
    @printf("    h    | error \n")
    @printf("---------|----------- \n")
    for p = 2:11
        h = 0.1^p
        dEh = zeros(dE)
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
        warn("""is seems the finite-difference test has failed, wich indicates
             that there is an inconsistency between the function and gradient
             evaluation""")
        return false
    end
end

"finite-difference test of a SitePotential V"
fdtest(V::SitePotential, X) =
    fdtest(x->at_energy(V, x), x->at_energy1(V, x), X)

end
