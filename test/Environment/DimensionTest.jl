#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 00:44:55
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "Dimension" begin
    @test EtherParticlesGPU.Environment.dimension(EtherParticlesGPU.Dimension1D) == 1
    @test EtherParticlesGPU.Environment.dimension(EtherParticlesGPU.Dimension2D) == 2
    @test EtherParticlesGPU.Environment.dimension(EtherParticlesGPU.Dimension3D) == 3
end
