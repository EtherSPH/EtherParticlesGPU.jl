#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/22 01:05:23
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractNeighbourSystem{IT <: Integer, FT <: AbstractFloat} end
