"""
## module DFT

### Summary

Provides Julia wrappers for DFT codes, principally GPAW at present
"""
module DFT

import JuLIP: AbstractCalculator, AbstractAtoms, positions, energy, site_energies

import JuLIP.ASE: AbstractASECalculator, ASEAtoms, rnn

export AbstractDFTCalculator,
       GPAWCalculator,
       WaveFunctions,
       Hamiltonian,
       Density,
       wavefunctions,
       hamiltonian,
       density,
       gd,
       finegd,
       kinetic_energy_density,
       potential_energy_density,
       energy_density,
       on_site_energies,
       site_energies

using PyCall

@pyimport ase.units as units
@pyimport gpaw
@pyimport gpaw.fd_operators as gpaw_fd_operators
@pyimport gpaw.io as gpaw_io

abstract AbstractDFTCalculator <: AbstractASECalculator
abstract WaveFunctions
abstract Density
abstract Hamiltonian

type GPAWCalculator <: AbstractDFTCalculator
  po::PyObject
end

type GPAWWaveFunctions <: WaveFunctions
  po::PyObject
end

type GPAWHamiltonian <: Hamiltonian
  po::PyObject
end

type GPAWDensity <: Density
  po::PyObject
end

typealias GPAWData Union{GPAWHamiltonian,GPAWDensity,GPAWWaveFunctions}

type GPAWGridDescriptor
  po::PyObject
end

GPAWCalculator(;kwargs...) = GPAWCalculator(gpaw.GPAW(;kwargs...))

wavefunctions(g::GPAWCalculator) = GPAWWaveFunctions(g.po[:wfs])
hamiltonian(g::GPAWCalculator) = GPAWHamiltonian(g.po[:hamiltonian])
density(g::GPAWCalculator) = GPAWDensity(g.po[:density])

gd(g::GPAWData) = GPAWGridDescriptor(g.po[:gd])
gd(g::GPAWCalculator) = gd(density(g))

finegd(g::GPAWData) = GPAWGridDescriptor(g.po[:finegd])
finegd(g::GPAWCalculator) = finegd(density(g))

"""
Helper function to look up array symbols in a Hamiltonian, Density
or WaveFunctions: return a PyArray that refers to the same data, not a copy.
"""
Base.getindex(g::GPAWData, s::Symbol) = PyArray(g.po[string(s)])

import Base.zeros
zeros(g::GPAWGridDescriptor, args...; kwargs...) =
    pycall(g.po[:zeros], PyArray, args...; kwargs...)

zeros(g::GPAWData, args...; kwargs...) = zeros(gd(g), args...; kwargs...)

function kinetic_energy_density(calc::AbstractCalculator, at::AbstractAtoms)
    energy(calc, at) # force a recalc if necessary
    kinetic_energy_density(wavefunctions(calc))
end

function potential_energy_density(calc::AbstractDFTCalculator, at::AbstractAtoms)
    energy(calc, at) # force a recalc if necessary
    potential_energy_density(hamiltonian(calc), density(calc))
end

"""
energy_density() is sum of kinetic_energy_density() over spins plus
potential_energy_density().
"""
energy_density(calc::AbstractDFTCalculator, at::AbstractAtoms) =
    squeeze(sum(kinetic_energy_density(calc, at), 1), 1) +
    potential_energy_density(calc, at)

function on_site_energies(calc::AbstractDFTCalculator, at::AbstractAtoms)
    energy(calc, at) # force a recalc if necessary
    on_site_energies(hamiltonian(calc), density(calc))
end

function site_energies(calc::AbstractDFTCalculator, at::AbstractAtoms, loc_func)
    E_a = zeros(length(at))
    E_a += partition(energy_density(calc, at), gd(calc), at, loc_func)
    E_a += on_site_energies(calc, at)
    E_a
end

"""
Compute DFT site energies using default localisation function

    ``\hat{X}(r) = exp(0.5*(r_cut - r0) / (|r| - r_cut))``

with `r0 = rnn(at)` and `r_cut = 2.5*r0`.
"""
function site_energies(calc::AbstractDFTCalculator, at::AbstractAtoms)
    r0 = rnn(at)
    r_cut = 2.5*r0
    site_energies(calc, at,
                  r -> exp(0.5*(r_cut - r0) / (r - r_cut)))
end

"""
    kinetic_energy_density(wfs::GPAWWaveFunctions)

Compute kinetic energy density `taut_sG` on coarse grid

    ``\tau(r) = 0.5 * \sum_{\sigma,k,n} f_{\sigma,k,n} | \nabla \psi_{\sigma,k,n}|^2``

# Arguments
* `wfs::WaveFunctions` - Wavefunctions of converged calculation. Only tested with
      finite difference mode (FDWaveFunctions).

# Returns
* `taut_sG::Array{Float64,4}`, shape [wfs.nspins] + wfs.gd.N_c
   Kinetic energy density for each spin on coarse grid
"""
function kinetic_energy_density(wfs::GPAWWaveFunctions)
    w = wfs.po
    g = gd(wfs)
    gradient_apply = [ gpaw_fd_operators.Gradient(g.po, v, n=3,
                        dtype=w[:dtype])[:apply] for v in 0:2 ]
    kpts = w[:kpt_u]
    #taut_sG = zeros(g, w[:nspins]) # FIXME better to alloc from Python?
    taut_sG = zeros([w[:nspins]; g.po[:N_c]]...) # alloc from Julia
    dpsit_G = zeros(g, dtype=w[:dtype])
    abs2_dpsit_G = zeros(g, dtype=w[:dtype])
    for kpt in kpts
        # FIXME: may need to load wavefunction from disk
        # psit_nG = kpt[:psit_nG][:__getitem__](pyslice)

        # FIXME should ideally use reference not copy, but slices
        # don't yet seem to be supported for PyArray
        psit_nG = kpt[:psit_nG]

        # apply gradient operator to wavefunctions and compute KE density
        # \tau(r) = 0.5 * \sum_{\sigma,k,n} f_{\sigma,k,n} | \nabla \psi_{\sigma,k,n}|^2
        f = kpt[:f_n]
        for n in eachindex(f)
            psit_G = psit_nG[n,:,:,:]
            for v in 1:3
                gradient_apply[v](psit_G, dpsit_G, kpt[:phase_cd])
                taut_sG[kpt[:s]+1,:,:,:] .+= 0.5 * f[n] * abs2.(dpsit_G)
                #axpy!()
            end
        end
    end
    return taut_sG
end

"""
    potential_energy_density(H::GPAWHamiltonian, rho::GPAWDensity)

Compute potential energy density on coarse grid

    ``e(r) =  \bar{e}(r) + e_{xc}(r) + e_H(r)``

Total energy is ``E = \int tau(r) + e(r) dr + \sum_a E_a``.

# Arguments
  * `H::GPAWHamiltonian``
        Hamiltonian containing results of converged calculation
  * `rho::GPAWDensity``
        Density containing results of converged calculation
  * `wfs::GPAWWaveFunctions`
        Wavefunctions of converged calculation. Only tested with
        finite difference mode (FDWaveFunctions).
# Returns
  *  `eden_G::Array{Float64,3}`, shape density.gd.N_c
       potential energy density e(r) on coarse grid
"""
function potential_energy_density(H::GPAWHamiltonian, rho::GPAWDensity)
    H.po[:vext] == nothing || error("no support for external potentials yet")

    # Check for corrections due to non-local XC
    Ekin_nl = H.po[:xc][:get_kinetic_energy_correction]()
    Ekin_nl == 0.0 || error("don't know how to partition non-local XC energies")

    # accumulate energy density on a real space coarse grid
    eden_G = zeros(gd(H))

    # Local pseudopotential energy on fine grid
    vbar_nt_g = H[:vbar_g] .* rho[:nt_g]
    vbar_nt_G = H.po[:restrict](vbar_nt_g)
    eden_G .+= vbar_nt_G

    # set effective potential \tilde{V} to vbar_g
    vt_sg = H[:vt_sg] # reference, not copy
    vt_sg[:,:,:,:] = 0.0
    for s in 1:H.po[:nspins]
        vt_sg[s,:,:,:] = H[:vbar_g]
    end

    # Evaluate Exchange-Correlation energy on fine grid
    exc_g = zeros(finegd(H))
    H.po[:xc][:calculate](finegd(H).po,
                          rho[:nt_sg], vt_sg, exc_g)
    exc_G = H.po[:restrict](exc_g)
    eden_G .+= exc_G

    # Evaluate Hartree potential on fine grid
    vHt_rhot_g = 0.5 * (H[:vHt_g] .* rho[:rhot_g])
    vHt_rhot_G = H.po[:restrict](vHt_rhot_g)
    eden_G .+= vHt_rhot_G

    return eden_G
end

"""
    on_site_energies(H::GPAWHamiltonian, rho::GPAWDensity)

on-site contribution to the site energies

# Arguments
*  `H::GPAWHamiltonian``
      Hamiltonian containing results of converged calculation
* `rho::GPAWDensity``
      Density containing results of converged calculation

# Returns
*  `Eonsite_a::array, shape len(atoms)``
      on-site contributions to total energy for each atom
"""
function on_site_energies(H::GPAWHamiltonian, rho::GPAWDensity)

    natoms = H.po[:atom_partition][:natoms]
    Ekin_a = zeros(natoms)
    Ebar_a = zeros(natoms)
    Epot_a = zeros(natoms)
    Exca_a = zeros(natoms)

    # Calculate atomic hamiltonians
    W_aL = Dict()
    for a in keys(rho.po[:D_asp])
        W_aL[a+1] = zeros((H.po[:setups][a+1][:lmax] + 1)^2)
    end
    rho.po[:ghat][:integrate](H[:vHt_g], W_aL)

    for (a, D_sp) in rho.po[:D_asp]
        a += 1 # convert from zero- to one-based indices
        W_L = W_aL[a]
        setup = H.po[:setups][a]

        D_p = sum(D_sp[1:H.po[:nspins],:], 1)[:]
        dH_p = (setup[:K_p] + setup[:M_p] +
                setup[:MB_p] + 2.0*setup[:M_pp]*D_p +
                setup[:Delta_pL] * W_L)

        Ekin_a[a] = dot(setup[:K_p], D_p) + setup[:Kc]
        Ebar_a[a] = setup[:MB] + dot(setup[:MB_p], D_p)
        Epot_a[a] = setup[:M] + dot(D_p, setup[:M_p] + setup[:M_pp] * D_p)

        dH_sp = similar(D_sp)
        dH_sp[1:H.po[:nspins],:] += dH_p'
        # We are not yet done with dH_sp; still need XC correction below
    end

    Ddist_asp = H.po[:atomic_matrix_distributor][:distribute](rho.po[:D_asp])
    for (a, D_sp) in Ddist_asp
        setup = H.po[:setups][a+1]
        dH_sp = similar(D_sp)
        Exca_a[a+1] = H.po[:xc][:calculate_paw_correction](setup, D_sp, dH_sp, a=a)
    end

    return Ekin_a + Epot_a + Ebar_a + Exca_a
end


"""
    partition(eden_G, grid, at, loc_func)

Partition energy density into site energies

    ``E_a = \int X_a(r) e(r) dr``

where ``X_a(r) = \hat{X}(r - r_a) / \bar{X}(r)``,
``\bar{X}(r) = \sum_a \hat{X}(r - r_a)`` and ``\hat{X}(r)`` is given by loc_func.

# Arguments
* eden_G::Array{Float64,3}, shape density.gd.N_c
    Energy density on coarse grid, as computed by `energy_density()`
* grid::GPAWGridDescriptor
    GridDescriptor uset to get grid point distance vectors
* at::AbstractAtoms
    Ionic configuration. Used to determine distance from grid points
    to each atom.
* loc_func
    Localisation function ``\hat{X}(r)`` defined for scalar distance r.
"""
function partition(eden_G::Array{Float64,3},
                   grid::GPAWGridDescriptor,
                   at::AbstractAtoms, loc_func)
    Xhat_aG = zeros(grid, length(at))
    X = positions(at)
    r0 = rnn(at)
    for a in 1:length(at)
        # distances from atom a to each of the coarse grid points, in Angstrom
        R = grid.po[:get_grid_point_distance_vectors](X[a]/units.Bohr)*units.Bohr
        r = squeeze(sqrt.(sum(R .^ 2, 1)), 1) # FIXME remove squeeze?
        Xhat_aG[a,:,:,:] = broadcast(loc_func, r)
    end

    Xbar_g = sum(Xhat_aG, 1)
    X_aG = Xhat_aG ./ Xbar_g

    E_a = zeros(length(at))
    for a in 1:length(at)
        E_a[a] = grid.po[:integrate](X_aG[a,:,:,:], eden_G)
    end
    E_a
end

end
