using JuLIP, ForwardDiff, StaticArrays, ReverseDiff, BenchmarkTools

# simple EAM potential
f(R) = sqrt( 1.0 + sum( exp(-norm(r)) for r in R ) )


nneigs = 30
R0 = [ @SVector rand(3) for n = 1:nneigs ]

f_fd(R) =  ForwardDiff.gradient( T -> f(vecs(T)),  mat(R)[:] ) |> vecs

function f_d(R::JVecsF)
   ∇f = zeros(JVecF, length(R))
   ρ̄ = sum( exp(-norm(r)) for r in R )
   dF = 0.5 * (1.0 + ρ̄)^(-0.5)
   return [ dF * (- exp(-norm(r))) * r/norm(r)  for r in R ]
end

@assert f_d(R0) ≈ f_fd(R0)

@btime f($R0);
@btime f_d($R0);
@btime f_fd($R0);



f1(R::AbstractMatrix) =
      sqrt( 1.0 + sum( exp.(-sqrt.(sum(abs2, R, 1))) ) )
f1_rd(R::Matrix) =
      reshape(
            ReverseDiff.gradient( S -> f1(reshape(S, 3, length(S) ÷ 3)), R[:] ),
            size(R) )
R1 = mat(R0)
f1_rd(R1)

@btime f1($R1);
@btime f1_rd($R1);
