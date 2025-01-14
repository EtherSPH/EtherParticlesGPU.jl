#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 00:42:38
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "Core" begin
    include("DimensionTest.jl")
    include("ParallelTest.jl")
    include("DomainTest.jl")
    include("ParameterTest.jl")
end
