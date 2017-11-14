using BenchmarkTools
using JuLIP, JuLIP.ASE, JuLIP.Potentials

println()
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("   JuLIP Performance Regression Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println()

println("--------------------------------------------------------------------------")
println("LENNARD-JONES")
const lj = lennardjones(r0=JuLIP.ASE.rnn("Al"))
const al = bulk("Al", cubic=true) * (10,10,8)

print("Energy Assembly (without nlist): ")
@btime energy($lj, $al)

print("Energy Assembly (with nlist):    ")
@btime energy($lj,  rattle!($al, 0.001))

print("Force Assembly (without nlist):  ")
@btime forces($lj, $al)

print("Force Assembly (with nlist):     ")
@btime forces($lj,  rattle!($al, 0.001))


# println("--------------------------------------------------------------------------")
# println("EAM (Splines)")
# data = joinpath(dirname(@__FILE__), "..", "data") * "/"
# const eam_Fe = JuLIP.Potentials.EAM(data * "pfe.plt", data * "ffe.plt", data * "F_fe.plt")
# const fe = bulk("Fe", cubic=true) * (12,12,8)
#
# print("Energy Assembly (without nlist): ")
# @btime energy($eam_Fe, $fe)
#
# print("Energy Assembly (with nlist):    ")
# f = () -> energy(eam_Fe,  rattle!(fe, 0.001))
# @btime f()
#
#
# println("--------------------------------------------------------------------------")
# println("STILLINGER-WEBER")
# const sw = StillingerWeber()
# const si = bulk("Si", cubic=true) * (12,15,12)
#
# print("Energy Assembly (without nlist): ")
# @btime energy($sw, $si)
#
# print("Energy Assembly (with nlist):    ")
# f = () -> energy(sw,  rattle!(si, 0.001))
# @btime f()
