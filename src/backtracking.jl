
import Optim, LineSearches


# this is a simplified version of the code in `LineSearches.jl`,
# but with the added feature that it provides a maximal step-length

function julipbt!{T}(df,
                    x::Vector{T},
                    s::Vector,
                    x_scratch::Vector,
                    gr_scratch::Vector,
                    lsr::LineSearches.LineSearchResults,
                    alpha::Real = 1.0,
                    mayterminate::Bool = false,
                    c1::Real = 1e-4,
                    rhohi::Real = 0.5,
                    rholo::Real = 0.1,
                    iterations::Integer = 1_000,
                    maxstep = Inf)
    # Count the total number of iterations
    iteration = 0
    # Count number of parameters
    n = length(x)
    # read f_x and slope from LineSearchResults
    f_x = lsr.value[end]
    gxp = lsr.slope[end]
    # Tentatively move a distance of alpha in the direction of s
    @simd for i in 1:n
        @inbounds x_scratch[i] = x[i] + alpha * s[i]
    end
    push!(lsr.alpha, alpha)
    # Backtrack until we satisfy sufficient decrease condition
    f_x_scratch = NLSolversBase.value!(df, x_scratch)
    push!(lsr.value, f_x_scratch)
    while f_x_scratch > f_x + c1 * alpha * gxp
        # Increment the number of steps we've had to perform
        iteration += 1
        # Ensure termination
        if iteration > iterations
            throw(LineSearchException("Linesearch failed to converge, reached maximum iterations $(iterations).", lsr.alpha[end], lsr))
        end
        # Shrink proposed step-size:
        alphatmp = - (gxp * alpha^2) / ( 2.0 * (f_x_scratch - f_x - gxp*alpha) )
        alphatmp = min(alphatmp, alpha*rhohi) # avoid too small reductions
        alpha = max(alphatmp, alpha*rholo) # avoid too big reductions
        # shrink it some more: want |alpha * s|_inf ≦ maxstep; i.e. alpha ≦ maxstep / |s|_inf
        alpha = min(alpha, maxstep / vecnorm(s))
        # store the step
        push!(lsr.alpha, alpha)
        # Update proposed position
        @simd for i in 1:n
            @inbounds x_scratch[i] = x[i] + alpha * s[i]
        end
        # Evaluate f(x) at proposed position
        f_x_scratch = NLSolversBase.value!(df, x_scratch)
        push!(lsr.value, f_x_scratch)
    end
    return alpha
end
