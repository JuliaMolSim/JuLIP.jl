
# try to install ase if it isn't yet installed
try
   @pyimport ase as __ase__
   _found_ase_ = true
catch
   warn("""JuLIP was unable to import `ase`. This suggests that
           `ase` is not installed in the Python distribution currently
           used by `PyCall`. Most likely, this can be fixed by calling
            `JuLIP.install_ase()`. If this fails, or if you don't want an
            automatic installation then please follow
            the instructions at `https://wiki.fysik.dtu.dk/ase/install.html`""")
end


# try to install ase if it isn't yet installed
try
   @pyimport matscipy as __matscipy__
   _found_matscipy_ = true
catch
   warn("""JuLIP was unable to import `matscipy`. This suggests that
           `matscipy` is not installed in the Python distribution currently
           used by `PyCall`. Most likely, this can be fixed by calling
            `JuLIP.install_matscipy()`. If this fails, or if you don't want an
            automatic installation then please follow
            the instructions at ``""")
end


function install_ase()
   pipcmd = `$(PyCall.pyprogramname[1:end-6])pip install --upgrade --user ase`
   run(`$(pipcmd)`)
end

function install_matscipy()
   pipcmd = `$(PyCall.pyprogramname[1:end-6])pip install --upgrade --user matscipy`
   run(`$(pipcmd)`)
end
