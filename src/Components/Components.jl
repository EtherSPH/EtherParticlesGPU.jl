#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 16:36:03
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

module Components

using EtherParticlesGPU.Environment

include("Parameter.jl")
export AbstractParameter
export AbstractParameter2D, AbstractParameter3D

end
