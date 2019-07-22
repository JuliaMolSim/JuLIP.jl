"""
`module Experimental`

TODO: write documentation
"""
module Experimental

using JuLIP: AbstractAtoms, dofs, set_dofs!, atomdofs
using JuLIP.Potentials
using ForwardDiff

export newton!, constrained_bond_newton!

function newton(x, g, h; maxsteps=10, tol=1e-5)
    for i in 1:maxsteps
        gi = g(x)
        norm(gi, Inf) < tol && break
        x = x - h(x) \ gi
        @show i, norm(gi, Inf)
    end
    return x
end

function newton!(a::AbstractAtoms; maxsteps=20, tol=1e-10)
    x = newton(dofs(a),
               x -> gradient(a, x),
               x -> hessian(a, x))
    set_dofs!(a, x)
    return x
end

function constrained_bond_newton!(a::AbstractAtoms, i::Integer, j::Integer,
                                  bondlength::AbstractFloat;
                                  maxsteps=20, tol=1e-10)
    I1 = atomdofs(a, i)
    I2 = atomdofs(a, j)
    I1I2 = [I1; I2]

    # bondlength of target bond
    blen(x) = norm(x[I2] - x[I1])

    # constraint function
    c(x) = blen(x) - bondlength

    # gradient of constraint
    function dc(x)
        r = zeros(length(x))
        r[I1] = (x[I1]-x[I2])/blen(x)
        r[I2] = (x[I2]-x[I1])/blen(x)
        return r
    end

    # define some auxiliary index arrays that start at 1 to
    # allow the ForwardDiff hessian to be done only on relevent dofs, x[I1I2]
    _I1 = 1:length(I1)
    _I2 = length(I1)+1:length(I1)+length(I2)
    _blen(x) = norm(x[_I2] - x[_I1])
    _c(x) = _blen(x) - bondlength

    function ddc(x)
        s = spzeros(length(x), length(x))
        s[I1I2, I1I2] = ForwardDiff.hessian(_c, x[I1I2])
        return s
    end

    # Define Lagrangian L(x, λ) = E - λC and its gradient and hessian
    L(x, λ) = energy(a, x) - λ*c(x)
    L(z) = L(z[1:end-1], z[end])

    dL(x, λ) = [gradient(a, x) - λ * dc(x); -c(x)]
    dL(z) = dL(z[1:end-1], z[end])

    ddL(x, λ) = [(hessian(a, x) - λ * ddc(x)) -dc(x); -dc(x)' 0.]
    ddL(z) = ddL(z[1:end-1], z[end])

    # Use Newton scheme to find saddles L where ∇L = 0
    z = [dofs(a); -10.0*c(dofs(a))]
    z = newton(z, dL, ddL, maxsteps=maxsteps, tol=tol)
    set_dofs!(a, z[1:end-1])
    return z
end

end
