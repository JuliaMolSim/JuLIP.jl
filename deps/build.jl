
println("Installing Dependencies of `JuLIP.jl`: `ase` and `matscipy`")

try
   @pyimport ase as _ase_
catch
   println("""`ase` was not found, trying to install via pip. If this fails,
            please file an issue and try to install it manually, following
            the instructions at `https://wiki.fysik.dtu.dk/ase/install.html`""")
   pipcmd = `$(PyCall.pyprogramname[1:end-6])pip install --upgrade --user ase`
   run(`$(pipcmd)`)
end

try
   @pyimport matscipy as _matscipy_
catch
   println("""`ase` was not found, trying to install it. If this fails,
            please file an issue and try to install it manually, following
            the instructions at `https://github.com/libAtoms/matscipy`""")
   run(`git clone https://github.com/libAtoms/matscipy.git`)
   run(`cd matscipy`)
   run(`$(PyCall.pyprogramname) setup.py build`)
   run(`$(PyCall.pyprogramname) setup.py install`)
   run(`cd ..`)
   run(`rm -rf matscipy`)
end
