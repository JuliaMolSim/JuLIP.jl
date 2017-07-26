
## Installing numba and llvmlite correctly

This will probably be needed to get `chemview` to run, though at the moment
it seems `ipywidgets` is the real culprit and there is little hope for
a quick fix.

```
git clone https://github.com/numba/llvmlite
cd llvmlite
LLVM_CONFIG=/Users/ortner/gits/julia/usr/tools/llvm-config python setup.py install
LLVM_CONFIG=/Users/ortner/gits/julia/usr/tools/llvm-config pip install numba
```
then start `ipython` and check that `import numba` works, then start Julia
and check that `using PyCall; @pyimport numba` works as well.


## (Old) Alternative Installation Instructions for `imolecule`

If Option 1 fails, then try the following instructions, which
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

<!--
(Update: the configuration can be written directly to a JSON file, which
ought to circumvent the need for OpenBabel. Need to test this on a clean system.)
-->
