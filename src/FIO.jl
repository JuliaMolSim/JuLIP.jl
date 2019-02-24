
module FIO

export load_json, save_json, decode_dict

#######################################################################
#                     Conversions to and from Dict
#######################################################################

# eventually we want to be able to serialise all JuLIP types to Dict 
# and back and those Dicts may only contain elementary data types 
# this will then allow us to load them back via decode_dict without 
# having to know the type in the code

"""
`decode_dict(D::Dict) -> ?`

Looks for a key `__id__` in `D` whose value is used to dynamically dispatch 
the decoding to 
```julia
convert(Val(Symbol(D["__id__"])))
```
That is, a user defined type must implement this `convert(::Val, ::Dict)` 
utility function. The typical code would be 
```julia
module MyModule
    struct MyStructA
        a::Float64 
    end 
    Dict(A::MyStructA) = Dict( "__id__" -> "MyModule_MyStructA", 
                               "a" -> a )
    MyStructA(D::Dict) = A(D["a"])
    Base.convert(::Val{:MyModule_MyStructA})
end
```
The user is responsible for choosing an id that is sufficiently unique. 

The purpose of this function is to enable the loading of more complex JSON files 
and automatically converting the parsed Dict into the appropriate composite 
types. It is intentional that a significant burden is put on the user code 
here. This will maximise flexibiliy, e.g., by introducing version numbers, 
and being able to read old version in the future.
"""
function decode_dict(D::Dict) 
    if !haskey(D, "__id__")
        @error("JuLIP.IO.decode_dict: `D` has no key `__id__`")
    end 
    return convert(Val(Symbol(D["__id__"])), D)
end 
    
#######################################################################
#                     JSON
#######################################################################

using JSON

function load_json(fname::AbstractString)
    return JSON.parsefile(fname)
end
    
function save_json(fname::AbstractString, D::Dict; indent=2)
    f = open(fname, "w")
    JSON.print(f, D, indent)
    close(f)
    return nothing
end
     
end 