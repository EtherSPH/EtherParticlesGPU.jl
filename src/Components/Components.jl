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

# once I decide to parse a struct onto GPU
# later I find it's more easy to parse a NamedTuple onto GPU
include("NamedIndex/NamedIndex.jl")
export NamedIndex
export get_index_named_tuple
export get_int_n_capacity, get_float_n_capacity

end
