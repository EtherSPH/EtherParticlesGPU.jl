#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 14:13:07
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

include("../../oneapi_head.jl")
include("shared.jl")

using Atomix

struct Particles
    n_particles::oneArray{IT, 1} # (1,)
    tags::oneArray{IT, 1} # (n_particles,)
    positions::oneArray{FT, 2} # (n_particles, 2)
    accumulated_positions::oneArray{FT, 2} # (n_particles, 2)
    neighbour_count::oneArray{IT, 1} # (n_particles,)
    neighbour_index::oneArray{IT, 2} # (n_particles, max_neighbour_count)
    neighbour_relative_positions::oneArray{FT, 2} # (n_particles, max_neighbour_count * 2)
end

@kernel function device_findneighbours!(n_particles, positions, neighbour_count, neighbour_index, neighbour_relative_positions)
    I = KernelAbstractions.@index(Global)
    @inbounds x0 = positions[I, 1]
    @inbounds y0 = positions[I, 2]
    J = 1
    @inbounds while J <= n_particles[1]
        x1 = positions[J, 1]
        y1 = positions[J, 2]
        dx = x1 - x0
        dy = y1 - y0
        r2 = dx * dx + dy * dy
        if r2 < kR2
            neighbour_count[I] += 1
            ni = neighbour_count[I]
            neighbour_index[I, ni] = J
            neighbour_relative_positions[I, ni * 2 - 1] = dx
            neighbour_relative_positions[I, ni * 2] = dy
        end
        J += 1
    end
end

@kernel function device_action!(tags, accumulated_positions, neighbour_count, neighbour_relative_positions)
    I = KernelAbstractions.@index(Global)
    @inbounds tag_I = tags[I]
    
end