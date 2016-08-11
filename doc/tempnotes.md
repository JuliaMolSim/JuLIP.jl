
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
