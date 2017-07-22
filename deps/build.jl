using PyCall


println("Installing Dependencies of `JuLIP.jl`: `ase` and `matscipy`")

if is_unix()
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
      println("""`matscipy` was not found, trying to install it. If this fails,
               please file an issue and try to install it manually, following
               the instructions at `https://github.com/libAtoms/matscipy`""")
      run(`pwd`)
      run(`git clone https://github.com/libAtoms/matscipy.git`)
      cd(joinpath(dirname(@__FILE__), "matscipy"))
      run(`$(PyCall.pyprogramname) setup.py build`)
      run(`$(PyCall.pyprogramname) setup.py install`)
      cd(dirname(@__FILE__))
      run(`rm -rf matscipy`)
   end

else
   println("""it looks like this is a windows machine? I don't dare try to
            automatically build the dependencies here -- sorry!""")
end
