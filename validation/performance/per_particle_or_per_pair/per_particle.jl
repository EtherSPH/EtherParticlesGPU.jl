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

using MLStyle

@inline function f11!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 1
        accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 1
        step += 1
    end
end

@inline function f12!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 1
        accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 2
        step += 1
    end
end

@inline function f13!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 1
        accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 3
        step += 1
    end
end

@inline function f21!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 2
        accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 1
        step += 1
    end
end

@inline function f22!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 2
        accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 2
        step += 1
    end
end

@inline function f23!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 2
        accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 3
        step += 1
    end
end

@inline function f31!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 3
        accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 1
        step += 1
    end
end

@inline function f32!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 3
        accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 2
        step += 1
    end
end

@inline function f33!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 3
        accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 3
        step += 1
    end
end

struct Particles
    n_particles::AbstractArray{IT, 1} # (1,)
    tags::AbstractArray{IT, 1} # (n_particles,)
    positions::AbstractArray{FT, 2} # (n_particles, 2)
    accumulated_positions::AbstractArray{FT, 2} # (n_particles, 2)
    neighbour_count::AbstractArray{IT, 1} # (n_particles,)
    neighbour_index::AbstractArray{IT, 2} # (n_particles, max_neighbour_count)
    neighbour_relative_positions::AbstractArray{FT, 2} # (n_particles, max_neighbour_count * 2)
end

function Particles(n)
    n_particles = [n]
    tags = zeros(IT, n)
    positions = zeros(FT, n, 2)
    accumulated_positions = zeros(FT, n, 2)
    neighbour_count = zeros(IT, n)
    neighbour_index = zeros(IT, n, kMaxNeighbourCount)
    neighbour_relative_positions = zeros(FT, n, kMaxNeighbourCount * 2)
    for I in 1:n
        positions[I, 1], positions[I, 2], tags[I] = get_particle(I)
    end
    n_particles = CT(n_particles)
    tags = CT(tags)
    positions = CT(positions)
    accumulated_positions = CT(accumulated_positions)
    neighbour_count = CT(neighbour_count)
    neighbour_index = CT(neighbour_index)
    neighbour_relative_positions = CT(neighbour_relative_positions)
    return Particles(
        n_particles,
        tags,
        positions,
        accumulated_positions,
        neighbour_count,
        neighbour_index,
        neighbour_relative_positions,
    )
end

@kernel function device_findneighbours!(
    n_particles,
    positions,
    neighbour_count,
    neighbour_index,
    neighbour_relative_positions,
)
    I = KernelAbstractions.@index(Global)
    @inbounds x0 = positions[I, 1]
    @inbounds y0 = positions[I, 2]
    J = 1
    # * 暴力循环
    @inbounds while J <= n_particles[1]
        x1 = positions[J, 1]
        y1 = positions[J, 2]
        dx = x1 - x0
        dy = y1 - y0
        r2 = dx * dx + dy * dy
        if r2 <= kR2 && I != J
            neighbour_count[I] += 1
            ni = neighbour_count[I]
            neighbour_index[I, ni] = J
            neighbour_relative_positions[I, ni * 2 - 1] = dx
            neighbour_relative_positions[I, ni * 2] = dy
        end
        J += 1
    end
end

@kernel function device_action!(
    tags,
    accumulated_positions,
    neighbour_count,
    neighbour_index,
    neighbour_relative_positions,
)
    I = KernelAbstractions.@index(Global)
    @inbounds tag_I = tags[I]
    NI = neighbour_count[I]
    NJ = 1
    @inbounds while NJ <= NI
        J = neighbour_index[I, NJ]
        tag_J = tags[J]
        @match (tag_I, tag_J) begin
            (kTag1, kTag1) => f11!(
                I,
                J,
                neighbour_relative_positions[I, NJ * 2 - 1],
                neighbour_relative_positions[I, NJ * 2],
                accumulated_positions,
            )
            (kTag1, kTag2) => f12!(
                I,
                J,
                neighbour_relative_positions[I, NJ * 2 - 1],
                neighbour_relative_positions[I, NJ * 2],
                accumulated_positions,
            )
            (kTag1, kTag3) => f13!(
                I,
                J,
                neighbour_relative_positions[I, NJ * 2 - 1],
                neighbour_relative_positions[I, NJ * 2],
                accumulated_positions,
            )
            (kTag2, kTag1) => f21!(
                I,
                J,
                neighbour_relative_positions[I, NJ * 2 - 1],
                neighbour_relative_positions[I, NJ * 2],
                accumulated_positions,
            )
            (kTag2, kTag2) => f22!(
                I,
                J,
                neighbour_relative_positions[I, NJ * 2 - 1],
                neighbour_relative_positions[I, NJ * 2],
                accumulated_positions,
            )
            (kTag2, kTag3) => f23!(
                I,
                J,
                neighbour_relative_positions[I, NJ * 2 - 1],
                neighbour_relative_positions[I, NJ * 2],
                accumulated_positions,
            )
            (kTag3, kTag1) => f31!(
                I,
                J,
                neighbour_relative_positions[I, NJ * 2 - 1],
                neighbour_relative_positions[I, NJ * 2],
                accumulated_positions,
            )
            (kTag3, kTag2) => f32!(
                I,
                J,
                neighbour_relative_positions[I, NJ * 2 - 1],
                neighbour_relative_positions[I, NJ * 2],
                accumulated_positions,
            )
            (kTag3, kTag3) => f33!(
                I,
                J,
                neighbour_relative_positions[I, NJ * 2 - 1],
                neighbour_relative_positions[I, NJ * 2],
                accumulated_positions,
            )
        end
        NJ += 1
    end
end

@inline function host_findneighbours!(particles::Particles)
    KernelAbstractions.fill!(particles.neighbour_count, IT(0))
    device_findneighbours!(Backend, kThreadNumber)(
        particles.n_particles,
        particles.positions,
        particles.neighbour_count,
        particles.neighbour_index,
        particles.neighbour_relative_positions,
        ndrange = (kParticleNumber,),
    )
    KernelAbstractions.synchronize(Backend)
end

@inline function host_action!(particles::Particles)
    device_action!(Backend, kThreadNumber)(
        particles.tags,
        particles.accumulated_positions,
        particles.neighbour_count,
        particles.neighbour_index,
        particles.neighbour_relative_positions,
        ndrange = (kParticleNumber,),
    )
    KernelAbstractions.synchronize(Backend)
end

function main()
    particles = Particles(kParticleNumber)
    @info "warm up"
    host_findneighbours!(particles)
    host_action!(particles)
    @info "start"
    @time begin
        for _ in 1:kOuterLoop
            host_findneighbours!(particles)
            host_action!(particles)
        end
    end
    println(particles.neighbour_count |> maximum)
end

main()
