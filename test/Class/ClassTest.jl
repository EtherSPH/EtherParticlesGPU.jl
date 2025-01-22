#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/23 03:34:46
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "Class" begin
    include("NeighbourSystem/NeighbourSystemTest.jl")
end
