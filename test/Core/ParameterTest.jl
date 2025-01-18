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
    @kwdef struct Parameter2D <: EtherParticlesGPU.AbstractParameter2D{IT, FT}
        density::FT = 1.0 |> FT
        sound_speed::FT = 2.0 |> FT
        viscosity::FT = 3.0 |> FT
        gamma::IT = 7 |> IT
    end
    parameter_2d = Parameter2D()
    @test parameter_2d.density ≈ 1.0f0
    @test parameter_2d.sound_speed ≈ 2.0f0
    @test parameter_2d.viscosity ≈ 3.0f0
    @test parameter_2d.gamma == 7
    @test EtherParticlesGPU.dimension(parameter_2d) == 2
end
