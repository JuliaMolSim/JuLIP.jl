# JuLIP: Julia Library for Interatomic Potentials

| **Build Status** | **Social** |
|:-:|:-:|
| [![Build Status][build-img]][build-url] | [![][gitter-img]][gitter-url] |

<!-- [![Build Status](https://travis-ci.org/libAtoms/JuLIP.jl.svg?branch=master)](https://travis-ci.org/libAtoms/JuLIP.jl) -->

A package for rapid implementation and testing of new interatomic potentials and
molecular simulation algorithms. Requires v0.5 or v0.6 of Julia. Current
development is for Julia v0.6.x. Documentation is essentially non-existent but
the inline documentations is reasonably complete.

The design of `JuLIP` is heavily inspired by [ASE](https://gitlab.com/ase/ase).
The main motivation for `JuLIP` is that, while `ASE` is pure Python and hence
relies on external software to efficiently evaluate interatomic potentials,
Julia allows the  implementation of fast potentials in pure Julia, often in just
a few lines of code. `ASE` bindings compatible with `JuLIP` are provided by
[ASE.jl](https://github.com/cortner/ASE.jl.git).

Contributions are welcome, especially for producing examples and tutorials. Any
questions or suggestions, please ask on [![][gitter-img]][gitter-url].



# Installation

Install JuLIP, from the Julia REPL:
```julia
Pkg.add("JuLIP")
```
and run
```
Pkg.test("JuLIP")
```
to make sure the installation succeeded. If a test fails, please open an issue.

Most likely you will also want to ASE bindings; please see
[ASE.jl](https://github.com/cortner/ASE.jl.git) for more detail.


<!-- ## `imolecule` and dependencies

This part can be skipped if no visualisation is required; `using JuLIP` will then
simply print a warning.

`JuLIP.Visualise` uses the Python module `imolecule` to visualise atomistic
configurations in an IPython notebook. Its main dependency is
[OpenBabel](http://openbabel.org/wiki/Main_Page). Most recently, this could be installed succesfully (from the bash) using
```bash
conda install -c openbabel openbabel
pip install imolecule
``` -->

# Examples

The following are some minimal examples to just get something to run.
More intersting examples will hopefully follow soon.


## Vacancy in a bulk Si cell

```julia
using JuLIP
at = bulk(:Si, cubic=true) * 4
deleteat!(at, 1)
set_calculator!(at, StillingerWeber())
minimise!(at)
# Visualisation is current not working
# JuLIP.Visualise.draw(at)   # (this will only work in a ipynb)
```
see the `BulkSilicon.ipynb` notebook under `examples` for an extended
example.

## Construction of a Buckingham potential

```julia
using JuLIP
r0 = rnn(:Al)
pot = let A = 4.0, r0 = r0
   @analytic r -> 6.0 * exp(- A * (r/r0 - 1.0)) - A * (r0/r)^6
end
pot = pot * SplineCutoff(2.1 * r0, 3.5 * r0)
# `pot` can now be used as a calculator to do something interesting ...
```

## Site Potential with AD

```julia
using JuLIP
# and EAM-like site potential
f(R) = sqrt( 1.0 + sum( exp(-norm(r)) for r in R ) )
# wrap it into a site potential type => can be used as AbstractCalculator
V = ADPotential(f)
# evaluate V and ∇V
R0 = [ @SVector rand(3) for n = 1:nneigs ]
@show V(R0)
@show (@D V(R0))
```

<!-- ## An Example with TightBinding

**THIS IS PROBABLY BROKEN ON JULIA v0.6**

Similar to vacancy example but with a Tight-Binding Model. First install
`TightBinding.jl`:
```julia
Pkg.clone("https://github.com/ettersi/FermiContour.jl.git")
Pkg.clone("https://github.com/cortner/TightBinding.jl.git")
```
Then run
```julia
using JuLIP, TightBinding
TB = TightBinding
# sp model for Si (NRL-Tight Binding)
tbm = TB.NRLTB.NRLTBModel(elem=TB.NRLTB.Si_sp, nkpoints = (0,0,0))
# bulk crystal
at = bulk("Si", cubic=true) * 4
Eref = energy(tbm, at)
# create vacancy
deleteat!(at, 1)
Edef = energy(tbm, at)
# formation energy: (not really but sort of)
println("Vacancy formation energy = ", Edef - Eref * length(at)/(length(at)+1))
println("(probably this should not be negative! Increase simulation accuracy!)")
``` -->


[build-img]: https://travis-ci.org/libAtoms/JuLIP.jl.svg?branch=master
[build-url]: https://travis-ci.org/libAtoms/JuLIP.jl
[gitter-url]: https://gitter.im/libAtoms/JuLIP.jl
[gitter-img]: https://badges.gitter.im/libAtoms/JuLIP.jl.svg
