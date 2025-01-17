#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 14:13:26
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

include("../../oneapi_head.jl")
using Atomix

struct Particles
    n_particles::AbstractArray{IT, 1} # (1,)
    tags::AbstractArray{IT, 1} # (n_particles,)
    positions::AbstractArray{FT, 2} # (n_particles, 2)
    accumulated_positions::AbstractArray{FT, 2} # (n_particles, 2)
    neighbour_count::AbstractArray{IT, 1} # (n_particles,)
    neighbour_relative_positions::AbstractArray{FT, 2} # (n_particles, max_neighbour_count * 2)
end

struct PerPair
    n_pairs::AbstractArray{IT, 1} # (1,)
    pairs::AbstractArray{IT, 2} # (n_pairs, 2)
end

# @kernel function findneighbours!(n_particles, positions, neighbour_count, neighbour_relative_positions)
#     I = KernelAbstractions.@index(Global)
#     @inbounds x0 = positions[I, 1]
#     @inbounds y0 = positions[I, 2]
#     J = 1
#     @inbounds while J <= n_particles[1]
#         x1 = positions[J, 1]
#         y1 = positions[J, 2]
#         dx = x1 - x0
#         dy = y1 - y0
#         r2 = dx * dx + dy * dy
#         if r2 < kR2
#             neighbour_count[I] += 1
#             neighbour_relative_positions[I, neighbour_count[I] * 2 - 1] = dx
#             neighbour_relative_positions[I, neighbour_count[I] * 2] = dy
#         end
#         J += 1
#     end
# end
