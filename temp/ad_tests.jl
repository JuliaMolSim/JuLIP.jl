using JuLIP, ForwardDiff, StaticArrays, BenchmarkTools

# simple EAM potential
f(R) = sqrt( 1.0 + sum( exp(-norm(r)) for r in R ) )


nneigs = 30
R0 = [ @SVector rand(3) for n = 1:nneigs ]

f_fd(R) =  ForwardDiff.gradient( T -> f(vecs(T)),  mat(R)[:] ) |> vecs

function f_d(R::AbstractVector{<:JVec})
   ∇f = zeros(JVecF, length(R))
   ρ̄ = sum( exp(-norm(r)) for r in R )
   dF = 0.5 * (1.0 + ρ̄)^(-0.5)
   return [ dF * (- exp(-norm(r))) * r/norm(r)  for r in R ]
end

@assert f_d(R0) ≈ f_fd(R0)

@btime f($R0);
@btime f_d($R0);
@btime f_fd($R0);



# f1(R::AbstractMatrix) =
#       sqrt( 1.0 + sum( exp.(-sqrt.(sum(abs2, R, 1))) ) )
# f1_rd(R::Matrix) =
#       reshape(
#             ReverseDiff.gradient( S -> f1(reshape(S, 3, length(S) ÷ 3)), R[:] ),
#             size(R) )
# R1 = mat(R0)
# f1_rd(R1)
#
# @btime f1($R1);
# @btime f1_rd($R1);

##

module M
   using ForwardDiff
   myf(x) = x^2+x+1
   myg(x) = π   # deliberately incorrect gradient for testing

   function myf(d::ForwardDiff.Dual{T}) where T
      x = ForwardDiff.value(d)
      @show d
      # y = ( myf(x), myg(x) )
      ForwardDiff.Dual{T}( myf(x), myg(x) * ForwardDiff.partials(d) )
   end
end


using ForwardDiff
M.myf(1.0)
ForwardDiff.derivative(M.myf, 1.0)


##


module S
   using ForwardDiff, Dierckx

   modelspline() = Spline1D([0.0, 0.5, 1.0], rand(3); bc = "zero", k = 2)
   ev(s::Spline1D, x) = s(x)
   ev_d(s::Spline1D, x) = Dierckx.derivative(s, x, 1)
   ev_dd(s::Spline1D, x) = Dierckx.derivative(s, x, 2)

   function ev(s::Spline1D, d::ForwardDiff.Dual{T}) where T
      x = ForwardDiff.value(d)
      ForwardDiff.Dual{T}( s(x), ev_d(s, x) * ForwardDiff.partials(d) )
   end

   function ev_d(s::Spline1D, d::ForwardDiff.Dual{T}) where T
      x = ForwardDiff.value(d)
      ForwardDiff.Dual{T}( ev_d(s, x), ev_dd(s, x) * ForwardDiff.partials(d) )
   end
   # Dierckx.derivative(V.spl, r, nu)

   function compf(s, X)
      [ sum( ev_d(s, x)^p for x in X )  for p = 0:3 ]
   end

   function J_compf(s, X)
      [ (p * ev_d(s, X[i])^(p-1) * ev_dd(s, X[i])) for p=0:3, i=1:length(X) ]
   end
end

using ForwardDiff
s = S.modelspline()
ForwardDiff.derivative( x -> S.ev(s, x), 0.5) == S.ev_d(s, 0.5)
ForwardDiff.derivative( x -> S.ev_d(s, x), 0.5) == S.ev_dd(s, 0.5)

X = rand(5)
J = ForwardDiff.jacobian(X_ -> S.compf(s, X_), X)
J1 = S.J_compf(s, X)
J1 ≈ J



##

using JuLIP, Test, Printf, LinearAlgebra, SparseArrays, ForwardDiff
using JuLIP.Potentials, JuLIP.Testing

test_pots = joinpath(datadep"JuLIP_testpots", "JuLIP_data") * "/"
eam_Fe = JuLIP.Potentials.EAM(test_pots * "pfe.plt", test_pots * "ffe.plt", test_pots * "F_fe.plt")


##

at = set_pbc!( bulk(:Fe, cubic = true), false ) * 2
rattle!(at, 0.1)
nlist = neighbourlist(at, cutoff(eam_Fe))
_, Rs = neigs(nlist, 1)
Xs = [ [zero(JVecF)]; Rs ]
x0 = mat(Xs)[:]

function _conf2env(x)
   Xs = vecs(x)
   return [ Xs[j] - Xs[1] for j = 2:length(Xs) ]
end

_dV2conf(dV) = mat([[- sum(dV)]; dV])[:]

fsite = X -> JuLIP.evaluate(eam_Fe, _conf2env(X))
fsite_d = X -> _dV2conf( JuLIP.evaluate_d(eam_Fe, _conf2env(X)) )

@test fsite(x0) == JuLIP.evaluate(eam_Fe, Rs)
@test JuLIP.Testing.fdtest(fsite, fsite_d, x0)

# ------

fsite_dd = X -> ForwardDiff.jacobian(fsite_d, X)
JuLIP.Testing.fdtest_hessian(fsite_d, fsite_dd, x0)

H = fsite_dd(x0)
Hp = zeros(JMatF, 11, 11)
for i = 1:11, j = 1:11
   Hp[i, j] = JMatF( H[ i*3 .+ (1:3), j*3 .+ (1:3) ] ... )
end

H1 = JuLIP.Potentials.evaluate_dd(eam_Fe, Rs)
H1 ≈ Hp


##

at = set_pbc!( bulk(:Fe, cubic = true), false ) * 2
rattle!(at, 0.1)
set_calculator!(at, eam_Fe)
# hessian(eam_Fe, at)
println(@test fdtest(eam_Fe, at, verbose=true))
# println(@test fdtest_hessian( x->gradient(at, x), x->hessian(at, x), dofs(at) ))
# fdtest_hessian(x->gradient(at, x), x->hessian(at, x), dofs(at), verbose=true
#       )

H = hessian(eam_Fe, at)
fdtest_hessian(x->gradient(at, x), x->hessian(at, x), dofs(at), verbose=true
      )

x = dofs(at)
U = 0.1 * rand(JVecF, length(at))
u = mat(U)[:]
g1 = H * u
h = 1e-3
g2 = (gradient(at, x+1e-3 * u) - gradient(at, x-1e-3*u)) / (2*h)
g1 - g2


##
