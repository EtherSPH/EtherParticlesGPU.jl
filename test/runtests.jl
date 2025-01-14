#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 00:40:10
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

using Test
using KernelAbstractions
using EtherParticlesGPU

@testset "EtherParticlesGPU" begin
    include("Core/CoreTest.jl")
end
