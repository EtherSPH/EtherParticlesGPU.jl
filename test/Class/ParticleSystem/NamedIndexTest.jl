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
    # naming rule:
    # generallly rule: `prefix` + `name` + `suffix`
    # without causing ambiguity, the name should be as short as possible to improve both readability and coding-efficiency.
    # for example, `RVec` is better than `PositionVec`, `nRVec` is better than `neighbourRelativeVector`.
    # `Mass` is more detailed than simply one character `M`, `Density` is more specific than simply `rho` as `Rho` may be confused with other physical quantities.
    # 1. Usually, we use `CamelCase` to name a field.
    # 2. prefix `n` means the field is a field related to neighbour.
    # 3. prefix `d` means the field is a field related to derivative. for example, dDensity means ρ̇, dVelocityVec means v̇.
    # 4. when `n` and `d` are both needed, we use `nD` as prefix. for example, nDW means ∇Ẇ.
    # 5. suffix `Vec` means the field is a vector. for example, PositionVec means r, VelocityVec means v.
    # 6. suffix `Mat` means the field is a matrix. for example, StrainMat means ε, StressMat means σ.
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
