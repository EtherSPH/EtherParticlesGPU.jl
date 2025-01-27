#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/26 17:04:11
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

module Algorithm

using KernelAbstractions
using Atomix

using EtherParticlesGPU.Environment
using EtherParticlesGPU.Class

const kDefaultThreadNumber = 256

include("NeighbourSearch/NeighbourSearch.jl")

end
