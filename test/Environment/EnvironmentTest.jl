#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 16:57:40
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "Environment" begin
    include("DimensionTest.jl")
    include("ParallelTest.jl")
    include("DomainTest.jl")
end
