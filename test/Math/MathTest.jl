#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/27 03:13:02
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "Math" begin
    @testset "offset" begin
        @test EtherParticlesGPU.offset(1, 1) == 1
        @test EtherParticlesGPU.offset(1, 2) == 2
        @test EtherParticlesGPU.offset(1, 3) == 3
        @test EtherParticlesGPU.offset(5, 1) == 5
        @test EtherParticlesGPU.offset(5, 2) == 6
        @test EtherParticlesGPU.offset(5, 3) == 7
        @test EtherParticlesGPU.offset(1, 1, 3) == 1
        @test EtherParticlesGPU.offset(1, 2, 3) == 4
        @test EtherParticlesGPU.offset(1, 3, 3) == 7
        @test EtherParticlesGPU.offset(5, 1, 2) == 5
        @test EtherParticlesGPU.offset(5, 2, 2) == 7
    end
end
