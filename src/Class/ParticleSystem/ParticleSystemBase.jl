#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/24 19:38:29
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractParticleSystemBase{IT <: Integer, FT <: AbstractFloat} end
