#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 17:43:18
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

module Class

using KernelAbstractions
using EtherParticlesGPU.Environment

const kDefaultThreadNumber = 256
const kDefaultMaxNeighbourNumber = 50

include("NeighbourSystem/NeighbourSystem.jl")
include("ParticleSystem/ParticleSystem.jl")
export NamedIndex
export get_index_named_tuple
export get_int_n_capacity, get_float_n_capacity

end
