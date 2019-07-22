
using JuLIP, Test
using JuLIP.Testing


h3("Testing single `Atoms` <-> `Dict`")
at = bulk(:Cu, cubic=true) * 3
set_pbc!(at, (true, false, true))
rattle!(at, 0.1)
D = Dict(at)
at1 = Atoms(D)
println(@test(at == at1))
at2 = decode_dict(D)
println(@test(at == at2))

h3("Test JSON fio")
fn = tempname()
save_dict(fn, D)
D1 = load_dict(fn)
# D1 == D  => this will be false so don't test it!
at3 = decode_dict(D1)
println(@test at3 == at1)

h3("Test array of Atoms <-> Dict")
ats = [ (bulk(:Cu) * rand(2:4)) for n = 1:5 ]
Ds = Dict("ats" => Dict.(ats))
ats1 = decode_dict.(Ds["ats"])
println(@test ats1 == ats)

h3("Test JSON fio for array")
fn = tempname()
save_dict(fn, Ds)
Ds1 = load_dict(fn)
ats2 = decode_dict.(Ds1["ats"])
println(@test ats == ats2)
