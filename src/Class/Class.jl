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
export AbstractNeighbourSystem
export NeighbourSystem
export clean!
export AbstractPeriodicBoundaryPolicy
export NonePeriodicBoundaryPolicy
export PeriodicBoundaryPolicy2D, PeriodicBoundaryPolicy3D

include("ParticleSystem/ParticleSystem.jl")
export AbstractParticleSystem
export ParticleSystem
export get_n_particles, get_n_capacity
export toDevice!, toHost!

end
