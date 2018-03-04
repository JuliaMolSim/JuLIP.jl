

println("testing periodic notion of distance")
X1 = [ JVecF(0.0,0.0,0.0), JVecF(0.5,0.5,0.5)]
at = Atoms(:H, X1)
set_pbc!(at, (true, false, false))
set_cell!(at, [1.0 0.2 0.3; 0.0 1.0 0.1; 0.0 0.0 1.0])
X2 = [ X1[1], X1[2] + JVecF(1.0, 0.2, 0.3)]
X3 = [ X1[1] - JVecF(0.1, 0.0, 0.0), X1[2]]
X4 = [ X1[1], X1[2] + JVecF(0.0, 1.0, 0.1)]

@testset "dist" begin
   @test JuLIP.dist(at, X1, X2) < 1e-14
   @test JuLIP.dist(at, X1, X3) ≈ 0.1
   @test JuLIP.dist(at, X1, X4) ≈ norm(JVecF(0.0, 1.0, 0.1))
end
