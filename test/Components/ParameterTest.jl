#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 01:13:59
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "Parameter" begin
    IT = Int32
    FT = Float32
    @kwdef struct Parameter <: EtherParticlesGPU.AbstractParameter{IT, FT}
        density::FT = 1.0 |> FT
        sound_speed::FT = 2.0 |> FT
        viscosity::FT = 3.0 |> FT
        gamma::IT = 7 |> IT
    end
    parameter = Parameter()
    @test parameter.density ≈ 1.0f0
    @test parameter.sound_speed ≈ 2.0f0
    @test parameter.viscosity ≈ 3.0f0
    @test parameter.gamma == 7
end
