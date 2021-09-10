
module Deps 

using DataDeps

function fetch_test_pots() 
   register(DataDep(
         "JuLIP_testpots",
         "A few EAM potentials for testing",
         "https://www.dropbox.com/s/leub1c9ft1mm9fg/JuLIP_data.zip?dl=1",
         post_fetch_method = file -> run(`unzip $file`)
         ))
   return joinpath(datadep"JuLIP_testpots", "JuLIP_data") * "/"
end 


end