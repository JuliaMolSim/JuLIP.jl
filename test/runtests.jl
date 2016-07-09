using JuLIP
using Base.Test

# write your own tests here
# @test 1 == 1

X = rand(3, 3)
Y = pts(X) |> mat

@show X == Y
@show X === Y
Y[1] = -1.0
@show X[1]
