


hessian_pos(V::SitePotential, at::AbstractAtoms) =
      _precon_or_hessian_pos(V, at, evaluate_dd!)


# implementation of a generic assembly of a global block-hessian from
# local site-hessians
function _precon_or_hessian_pos(V::SitePotential, at::AbstractAtoms{T}, hfun) where {T}
   nlist = neighbourlist(at, cutoff(V))
   maxN = maxneigs(nlist)
   hEs = zeros(JMat{T}, maxN, maxN)
   tmp = alloc_temp_dd(V, maxN)
   I, J, Z = Int[], Int[], JMat{T}[]
   # a crude size hint
   for C in (I, J, Z); sizehint!(C, npairs(nlist)); end
   for (i, neigs, R) in sites(nlist)
      z = at.Z[neigs]
      z0 = at.Z[i]
      nneigs = length(neigs)
      # [1] the "off-centre" component of the hessian:
      # h[a, b] = ∂_{Ra} ∂_{Rb} V     (this is a nneigs x nneigs block-matrix)
      fill!(hEs, zero(JMat{T}))
      hfun(hEs, tmp, V, R, z, z0)
      for a = 1:nneigs, b = 1:nneigs
         # if norm(hEs[a,b], Inf) > 0
            push!(I, neigs[a])
            push!(J, neigs[b])
            push!(Z, hEs[a,b])
         # end
      end

      # [2] the ∂_{Ri} ∂_{Ra} terms
      # hib = ∂_{Ri} ∂_{Rb} V = - ∑_a ∂_{Ra} ∂_{Rb} V
      # also at the same time we pre-compute the centre-centre term:
      #    hii = ∂_{Ri} ∂_{Ri} V = - ∑_a ∂_{Ri} ∂_{Ra} V
      hii = zero(JMat{T})
      for b = 1:nneigs
         hib = -sum( hEs[a, b] for a = 1:nneigs )
         # if norm(hib, Inf) > 0
            hii -= hib
            append!(I, (i,         neigs[b] ))
            append!(J, (neigs[b],  i        ))
            append!(Z, (hib,       hib'     ))
         # end
      end

      # and finally add the  ∂_{Ri}^2 term, which is precomputed above
      # if norm(hii, Inf) > 0
         push!(I, i)
         push!(J, i)
         push!(Z, hii)
      # end
   end
   return sparse(I, J, Z, length(at), length(at))
end



# ================ AD and FD Hessians =============


_at2dofinds(i) = ((i-1) * 3) .+ (1:3)

function _coo_append!(I, J, Z, atinds, A)
   for i = 1:length(atinds), j = 1:length(atinds)
      idof = _at2dofinds(atinds[i])
      jdof = _at2dofinds(atinds[j])
      iA = _at2dofinds(i)
      jA = _at2dofinds(j)
      for a = 1:3, b = 1:3
         push!(I, idof[a])
         push!(J, jdof[b])
         push!(Z, A[iA[a], jA[b]])
      end
   end
   return nothing
end

function ad_hessian(V::SitePotential, at::Atoms{T}) where {T}
   # triplet format
   I, J, Z = Int[], Int[], T[]

   nlist = neighbourlist(at, cutoff(V))
   maxN = maxneigs(nlist)

   for (i0, neigs, Rs) in sites(nlist)
      Hsite = ad_site_hessian(V, Rs, at.Z, at.Z[i0])
      _coo_append!(I, J, Z, [ [i0]; neigs ], Hsite)
   end

   return sparse(I, J, Z, 3*length(at), 3*length(at))
end

function _conf2env(x)
   Xs = vecs(x)
   return [ Xs[j] - Xs[1] for j = 2:length(Xs) ]
end

_dV2conf(dV) = mat([[- sum(dV)]; dV])[:]

#   AN OLDER VERSION FOR COMPARISON WITH THE NEW ONE BELOW
# function ad_site_hessian(V::SimpleSitePotential, Rs, Zs, z0)
#    dVx = x -> _dV2conf(evaluate_d(V, _conf2env(x)))
#    return ForwardDiff.jacobian(dVx, [ zeros(3); mat(Rs)[:] ])
# end

# I think this form of evaluate_d automatically forwards to the above with SimpleSitePotential
function ad_site_hessian(V::SitePotential, Rs, Zs, z0)
   dVx = x -> _dV2conf(evaluate_d(V, _conf2env(x), Zs, z0))
   return ForwardDiff.jacobian(dVx, [ zeros(3); mat(Rs)[:] ])
end


# ====== TODO: revisit the FD hessians =========

# """
# `fd_hessian(V, R, h) -> H`
#
# If `length(R) = N` and `length(R[i]) = d` then `H` is an N × N `Matrix{SMatrix}` with
# each element a d × d static array.
# """
# function fd_hessian(V::SitePotential, R::Vector{SVec{D,T}}, h) where {D,T}
#    N = length(R)
#    H = zeros(SMat{D, D, T}, N, N)
#    return fd_hessian!(H, V, R, h)
# end
#
# """
# `fd_hessian!(H, V, R, h) -> H`
#
# Fill `H` with the hessian entries; cf `fd_hessian`.
# """
# function fd_hessian!(H, V::SitePotential, R::Vector{SVec{D,T}}, h) where {D,T}
#    N = length(R)
#    # convert R into a long vector and H into a big matrix (same part of memory!)
#    Rvec = mat(R)[:]
#    Hmat = zeros(N*D, N*D)   # reinterpret(T, H, (N*D, N*D))
#    # now re-define ∇V as a function of a long vector (rather than a vector of SVecs)
#    dV(r) = (evaluate_d(V, r |> vecs) |> mat)[:]
#    # compute the hessian as a big matrix
#    for i = 1:N*D
#       Rvec[i] +=h
#       dVp = dV(Rvec)
#       Rvec[i] -= 2*h
#       dVm = dV(Rvec)
#       Hmat[:, i] = (dVp - dVm) / (2 * h)
#       Rvec[i] += h
#    end
#    Hmat = 0.5 * (Hmat + Hmat')
#    # convert to a block-matrix
#    for i = 1:N, j = 1:N
#       Ii = (i-1) * D + (1:D)
#       Ij = (j-1) * D + (1:D)
#       H[i, j] = SMat{D,D}(Hmat[Ii, Ij])
#    end
#    return H
# end
#
# function fd_hessian(calc::AbstractCalculator, at::AbstractAtoms{T}, h) where {T}
#    N = length(at)
#    H = zeros(JMat{T}, N, N)
#    return fd_hessian!(H, calc, at, h)
# end
#
#
# """
# `fd_hessian!{D,T}(H, calc, at, h) -> H`
#
# Fill `H` with the hessian entries; cf `fd_hessian`.
# """
# function fd_hessian!(H, calc::AbstractCalculator, at::AbstractAtoms, h)
#    D = 3
#    N = length(at)
#    X = positions(at) |> mat
#    x = X[:]
#    # convert R into a long vector and H into a big matrix (same part of memory!)
#    Hmat = zeros(N*D, N*D)
#    # now re-define ∇V as a function of a long vector (rather than a vector of SVecs)
#    dE(x_) = (site_energy_d(calc, set_positions!(at, reshape(x_, D, N)), 1) |> mat)[:]
#    # compute the hessian as a big matrix
#    for i = 1:N*D
#       x[i] += h
#       dEp = dE(x)
#       x[i] -= 2*h
#       dEm = dE(x)
#       Hmat[:, i] = (dEp - dEm) / (2 * h)
#       x[i] += h
#    end
#    Hmat = 0.5 * (Hmat + Hmat')
#    # convert to a block-matrix
#    for i = 1:N, j = 1:N
#       Ii = (i-1) * D + (1:D)
#       Ij = (j-1) * D + (1:D)
#       H[i, j] = SMat{D,D}(Hmat[Ii, Ij])
#    end
#    return H
# end
