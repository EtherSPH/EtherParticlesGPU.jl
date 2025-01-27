#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/24 18:34:59
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "NamedIndex" begin
    IT = Int32
    dim = 2
    neighbour_count = 50
    int_named_tuple = (Tag = 1, nCount = 1, nIndex = 1 * neighbour_count)
    float_named_tuple = (
        RVec = dim,
        Mass = 1,
        Density = 1,
        Volume = 1,
        VelocityVec = dim,
        dVelocityVec = dim,
        dDensity = 1,
        Pressure = 1,
        StrainMat = dim * dim,
        dStrainMat = dim * dim,
        StressMat = dim * dim,
        nRVec = dim * neighbour_count,
        nR = neighbour_count,
        nW = neighbour_count,
        nDW = neighbour_count,
    )
    named_index = EtherParticlesGPU.Class.NamedIndex{IT}(int_named_tuple, float_named_tuple)
    @test EtherParticlesGPU.Class.get_int_n_capacity(named_index) == 1 + 1 + 1 * neighbour_count
    @test EtherParticlesGPU.Class.get_float_n_capacity(named_index) ==
          dim +
          1 +
          1 +
          1 +
          dim +
          dim +
          1 +
          1 +
          dim * dim +
          dim * dim +
          dim * dim +
          dim * neighbour_count +
          neighbour_count +
          neighbour_count +
          neighbour_count
end
