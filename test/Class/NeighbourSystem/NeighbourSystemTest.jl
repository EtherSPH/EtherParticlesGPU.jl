#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/23 03:34:23
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "NeighbourSystem" begin
    include("NeighbourSystemBaseTest.jl")
    include("ActivePairTest.jl")
end
