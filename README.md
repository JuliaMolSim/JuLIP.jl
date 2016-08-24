# JuLIP: Julia Library for Interatomic Potentials

<!-- [![Build Status](https://travis-ci.org/cortner/JuLIP.jl.svg?branch=master)](https://travis-ci.org/cortner/JuLIP.jl) -->

A package for rapid implementation and testing of new interatomic potentials and
molecular simulation algorithms. Requires v0.5 of Julia.

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

In the foreseeable future we plan to implement better optimised
calculators, create links to electronic structure packages, possibly
include potentials for molecules (the focus at the moment is materials).


<!-- The long-term vision for JuLIP is that it can be used in two ways: (1) as a
Julia version of ASE, using ASE in a minimal fashion; or (2) as a selection of
efficient calculators for ASE. -->



# Installation

JuLIP relies on [ASE](https://gitlab.com/ase/ase) and
 [matscipy](https://github.com/libAtoms/matscipy). These should be straightforward
to install from the shell:  (TODO: this actually might be false; need to try again on a clean system)
```bash
pip install ase
pip install matscipy
```
Afterwards, install JuLIP, from the Julia REPL:
```julia
Pkg.clone("https://github.com/yuyichao/FunctionWrappers.jl.git")
Pkg.clone("https://github.com/libAtoms/JuLIP.jl.git")
```
And run
```
Pkg.test("JuLIP")
```
to make sure the installation succeeded. Otherwise, open an issue.

## `imolecule` and dependencies

This part can be skipped if no visualisation is required.

`JuLIP.Visualise` uses the Python module `imolecule` to visualise atomistic
configurations in an IPython notebook. Its main dependency is
 [OpenBabel](http://openbabel.org/wiki/Main_Page). The following instructions
 were only tested on OS X.

To install `imolecule` simply type
```bash
pip install imolecule
```
in a terminal.

Installing OpenBabel is relatively straightforward; the key issue is to get
the Python bindings set up correctly, especially since there are normally
multiple Python versions on an OS X system (Apple, homebrew and anaconda).
The following instructions worked for 2.3.90, but make sure to change the SWIG version
number directories as needed:
```bash
conda install cmake lxml swig
git clone https://github.com/openbabel/openbabel.git
cd openbabel
cmake ../openbabel-2.3.2 -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON -DCMAKE_INSTALL_PREFIX=~/anaconda -DPYTHON_INCLUDE_DIR=~/anaconda/include/python2.7 -DCMAKE_LIBRARY_PATH=~/anaconda/lib -DSWIG_DIR=~/anaconda/share/swig/3.0.2/ -DSWIG_EXECUTABLE=~/anaconda/bin/swig -DPYTHON_LIBRARY=~/anaconda/lib/libpython2.7.so
make
make install
```
This should have installed all libraries and python packages in `~/anaconda`
and the data files in `/usr/local/share/openbabel/2.3.90/`. To tell `babel`
where to find them, the following lines must be added to `~/.bash_profile`:
```bash
export BABEL_DATADIR="/usr/local/share/openbabel/2.3.90/"
export BABEL_LIBDIR="/Users/ortner/anaconda/lib/openbabel/2.3.90/"
```

(Update: the configuration can be written directly to a JSON file, which
ought to circumvent the need for OpenBabel. Need to test this on a clean system.)


# Examples


## Vacancy in a bulk Si cell


```julia
using JuLIP
at = Atoms("Si", cubic=true, repeatcell=(4,4,4), pbc=(true,true,true))
deleteat!(at, 1)
set_calculator!(at, JuLIP.Potentials.StillingerWeber())
set_constraint!(at, FixedCell(at))
JuLIP.Solve.minimise!(at)
JuLIP.Visualise.show(at)   # (this will only work in a ipynb)
```
see the `BulkSilicon.ipynb` notebook under `examples` for an extended
example.


## Construction of a Buckingham potential

```julia
using JuLIP
using JuLIP.Potentials
r0 = JuLIP.ASE.rnn("Al")
A = 4.0;  # stiffness paramter
pot = AnalyticPotential( :( 6.0 * exp(- $A * (r/$r0 - 1.)) - $A * ($r0/r)^6 ) )
pot = pot * SplineCutoff(2.1*r0, 3.5*r0)   
calc = PairCalculator(pot)    # calculator to evaluate energies, forces, etc
# do something interesting ...
```
