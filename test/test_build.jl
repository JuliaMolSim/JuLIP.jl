

using Test, JuLIP

at = bulk(:Si)

at2 = copy(at)
@test at == at2
@test at !== at2
@test at.X === at2.X

at3 = deepcopy(at)
@test at == at3
@test at !== at3
@test at.X !== at3.X
@test at.X == at3.X

rot_at = rotate!(deepcopy(at), x=[1,1,1], y=[1,-1,0])
apply_defm!(rot_at, inv(rotation_matrix(x=[1,1,1], y=[1,-1,0])))
@test rot_at ≈ at