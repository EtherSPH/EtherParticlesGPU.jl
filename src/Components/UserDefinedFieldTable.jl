#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/22 01:14:34
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractUserDefinedFieldTable{IT <: Integer, DT <: Real} end
