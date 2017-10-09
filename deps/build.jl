using PyCall

function checkpip()
   # Import pip
   try
       @pyimport pip
   catch
       # If it is not found, install it
       println("""I couldn't find `pip`. I will try to download and install it
                  automatically, but if this fails, please install
                  manually, then try to build `JuLIP` again.""")
       get_pip = joinpath(dirname(@__FILE__), "get-pip.py")
       download("https://bootstrap.pypa.io/get-pip.py", get_pip)
       run(`$(PyCall.python) $get_pip --user`)
   end
end

function pip(pkgname)
   checkpip()
   pipcmd = `$(PyCall.pyprogramname[1:end-6])pip install --upgrade --user $(pkgname)`
   run(`$(pipcmd)`)
end



println("Installing Dependencies of `JuLIP.jl`: `ase` and `matscipy`")

if is_unix()
   try
      @pyimport ase as _ase_
   catch
      println("""`ase` was not found, trying to install via pip. If this fails,
               please file an issue and try to install it manually, following
               the instructions at `https://wiki.fysik.dtu.dk/ase/install.html`""")
      pip("ase")
   end

   try
      @pyimport matscipy as _matscipy_
   catch
      println("""`matscipy` was not found, trying to install it. If this fails,
               please file an issue and try to install it manually, following
               the instructions at `https://github.com/libAtoms/matscipy`""")
      pip("matscipy")
   end
else
   println("""it looks like this is a windows machine? I don't dare try to
            automatically build the dependencies here -- sorry!
            If installing them by hand turns out non-trivial, please file an
            issue.""")
end

# This here is a gist with a slightly different approach to installing
# the dependencies, using @pyimport pip
# https://gist.github.com/Luthaf/368a23981c8ec095c3eb
