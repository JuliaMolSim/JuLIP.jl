# JuLIP: Julia Library for Interatomic Potentials

[![Build Status](https://travis-ci.org/libAtoms/JuLIP.jl.svg?branch=master)](https://travis-ci.org/libAtoms/JuLIP.jl)

A package for rapid implementation and testing of new interatomic potentials and
molecular simulation algorithms. Requires v0.5 or v0.6 of Julia.

**NOTE ON v0.6:** The conversion to Julia v0.6 is fairly recent. It is likely there are
bugs, if you encounter a problem, please file an issue. The main change is
the construction of analytic pair potentials, which is now achieved with the
macro `@analytic`; see also the example below.

The structure of JuLIP is heavily inspired by [ASE](https://gitlab.com/ase/ase)
but uses more "Julian" notation.  JuLIP relies on ASE for much of the
materials modeling background such as generating computational cells for
different materials. The main idea for JuLIP is that, while ASE is pure Python and
hence relies on external software to efficiently evaluate interatomic potentials, Julia
allows the  implementation of fast potentials in pure Julia, often in just
a few lines of code.

At present, JuLIP is very much a work in progress. It provides
infrastructure to rapidly implement and test some simple potentials, and to
explore new molecular simulation algorithms.

Contributions are welcome.

<!-- In the foreseeable future we plan to implement better optimised
calculators, create links to electronic structure packages, possibly
include potentials for molecules (the focus at the moment is materials). -->
<!-- The long-term vision for JuLIP is that it can be used in two ways: (1) as a
Julia version of ASE, using ASE in a minimal fashion; or (2) as a selection of
efficient calculators for ASE. -->


# Installation

JuLIP relies on [ASE](https://gitlab.com/ase/ase) and
 [matscipy](https://github.com/libAtoms/matscipy). Most likely, they will
 be automatically installed the first time you import `JuLIP`. If not, then
 please follow the instructions on the respective websites.

Install JuLIP, from the Julia REPL:
```julia
Pkg.add("JuLIP")
```
and run
```
Pkg.test("JuLIP")
```
to make sure the installation succeeded. If a test fails, please open an issue.


## `imolecule` and dependencies

This part can be skipped if no visualisation is required; `using JuLIP` will then
simply print a warning.

`JuLIP.Visualise` uses the Python module `imolecule` to visualise atomistic
configurations in an IPython notebook. Its main dependency is
[OpenBabel](http://openbabel.org/wiki/Main_Page). Most recently, this could be installed succesfully (from the bash) using
```bash
conda install -c openbabel openbabel
pip install imolecule
```


<!-- ## Automatic Differentiation

There is some experimental AD support implemented; see `src/adsite.jl`, which
require `ForwardDiff` and `ReverseDiffPrototype`. These can be installed via
```
Pkg.add("ForwardDiff")
Pkg.clone("https://github.com/jrevels/ReverseDiffPrototype.jl.git")
```
If the packages are missing then the AD functionality is simply turned off. -->


# Examples

The following are some minimal examples to just get something to run.
More intersting examples will hopefully follow soon.


## Vacancy in a bulk Si cell

```julia
using JuLIP, JuLIP.ASE
at = bulk("Si", cubic=true) * 4
deleteat!(at, 1)
set_calculator!(at, JuLIP.Potentials.StillingerWeber())
set_constraint!(at, FixedCell(at))
JuLIP.Solve.minimise!(at)
JuLIP.Visualise.draw(at)   # (this will only work in a ipynb)
```
see the `BulkSilicon.ipynb` notebook under `examples` for an extended
example.

## Construction of a Buckingham potential

```julia
using JuLIP
using JuLIP.Potentials
r0 = JuLIP.ASE.rnn("Al")
pot = let A = 4.0, r0 = r0
   @analytic r -> 6.0 * exp(- A * (r/r0 - 1.0)) - A * (r0/r)^6
end
pot = pot * SplineCutoff(2.1 * r0, 3.5 * r0)
# `pot` can now be used as a calculator to do something interesting ...
```

## An Example with TightBinding

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
```
