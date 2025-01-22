#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 15:51:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@inline function neighbourCellCount(dim::Integer)::typeof(dim)
    return 3^dim # 9 for 2D, 27 for 3D
end

include("NeighbourSystemBase.jl")
include("ActivePair.jl")
include("PeriodicBoundary/PeriodicBoundary.jl")

"""
    - `base_`: the base of neighbour system
    - `active_pair_`: the active pair of neighbour system
    - `periodic_boundary_`: the periodic boundary of neighbour system
"""
abstract type AbstractNeighbourSystem{IT <: Integer, FT <: AbstractFloat} end

@inline function clean!(
    neighbour_system::AbstractNeighbourSystem{IT, FT},
)::Nothing where {IT <: Integer, FT <: AbstractFloat}
    KernelAbstractions.fill!(neighbour_system.base_.contained_particle_index_count_, IT(0))
    return nothing
end
