

using Test, JuLIP

if hasase
   println("Testing `jbulk == bulk` ...")
   for sym in [:Fe, :W, :Al, :Cu, :Si, :C, :Mg, :Be]
      @test isapprox(bulk(sym), Atoms(ASE.bulk(string(sym))), tol = 1e-12)
      @test isapprox(bulk(sym, cubic=true), Atoms(ASE.bulk(string(sym), cubic=true)), tol = 1e-12)
   end
end
