
println("-------------------")
println(" Testing JuLIP.ASE")
println("-------------------")

using JuLIP
at = Atoms("Al", cubic=true, repeat=(1,1,2))
@show positions(at) |> mat
