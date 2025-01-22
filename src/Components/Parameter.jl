#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 16:35:32
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

# AbstractParameter does not necessarily need to have `Dimension` type parameter.
# so I just simply remove it.
# Intger type and Float type are still kept.
abstract type AbstractParameter{IT <: Integer, FT <: AbstractFloat} end
