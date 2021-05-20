

using JuLIP, LinearAlgebra, DataDeps
using JuLIP.Potentials, JuLIP.Testing

# ##  EAM

# register(DataDep(
#     "JuLIP_testpots",
#     "A few EAM potentials for testing",
#     "https://www.dropbox.com/s/leub1c9ft1mm9fg/JuLIP_data.zip?dl=1",
#     post_fetch_method = file -> run(`unzip $file`)
#     ))

# test_pots = joinpath(datadep"JuLIP_testpots", "JuLIP_data") * "/"

# @info("Loading some interatomic potentials . .")
# calc = JuLIP.Potentials.EAM(test_pots * "pfe.plt",
#                               test_pots * "ffe.plt",
#                               test_pots * "F_fe.plt")

                              
# # EAM Potential
# at = set_pbc!( bulk(:Fe, cubic = true), false ) * 2

## Si  SW

calc = JuLIP.Potentials.StillingerWeber()
at = set_pbc!( bulk(:Si, cubic = true), false ) * 2

##

@info("With threads")
JuLIP.usethreads!(true)
fdtest(calc, at, verbose=true)

@info("Serial")
JuLIP.usethreads!(false)
fdtest(calc, at, verbose=true)
##

JuLIP.usethreads!(true)
Et = energy(calc, at) 
Ft = forces(calc, at)
Vt = virial(calc, at)

JuLIP.usethreads!(false)
Es = energy(calc, at) 
Fs = forces(calc, at)
Vs = virial(calc, at)

@info("Compare threaded vs serial values")

@show abs(Et - Es)
@show norm(Vt - Vs)
@show maximum(norm.(Ft - Fs))
