#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 14:13:26
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # ! 0.002040 seconds (1.54 k allocations: 40.781 KiB)
 =#

include("../../cuda_head.jl")
include("shared.jl")

using Atomix
using MLStyle

@inline function f11!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        Atomix.@atomic accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 1
        Atomix.@atomic accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 1
        step += 1
    end
end

@inline function f12!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        Atomix.@atomic accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 1
        Atomix.@atomic accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 2
        step += 1
    end
end

@inline function f13!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        Atomix.@atomic accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 1
        Atomix.@atomic accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 3
        step += 1
    end
end

@inline function f21!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        Atomix.@atomic accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 2
        Atomix.@atomic accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 1
        step += 1
    end
end

@inline function f22!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        Atomix.@atomic accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 2
        Atomix.@atomic accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 2
        step += 1
    end
end

@inline function f23!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        Atomix.@atomic accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 2
        Atomix.@atomic accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 3
        step += 1
    end
end

@inline function f31!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        Atomix.@atomic accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 3
        Atomix.@atomic accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 1
        step += 1
    end
end

@inline function f32!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        Atomix.@atomic accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 3
        Atomix.@atomic accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 2
        step += 1
    end
end

@inline function f33!(I, J, dx, dy, accumulated_positions)
    step = 1
    while step <= kInnerLoop
        Atomix.@atomic accumulated_positions[I, 1] += dx + accumulated_positions[J, 1] * 3
        Atomix.@atomic accumulated_positions[I, 2] += dy + accumulated_positions[J, 2] * 3
        step += 1
    end
end

struct Particles
    n_particles::AbstractArray{IT, 1} # (1,)
    tags::AbstractArray{IT, 1} # (n_particles,)
    positions::AbstractArray{FT, 2} # (n_particles, 2)
    accumulated_positions::AbstractArray{FT, 2} # (n_particles, 2)
end

function Particles(n)
    n_particles = [n] |> CT
    tags = zeros(IT, n)
    positions = zeros(FT, n, 2)
    accumulated_positions = zeros(FT, n, 2) |> CT
    for i in 1:n
        positions[i, 1], positions[i, 2], tags[i] = get_particle(i)
    end
    tags = tags |> CT
    positions = positions |> CT
    return Particles(n_particles, tags, positions, accumulated_positions)
end

struct PerPair
    n_pairs::AbstractArray{IT, 1} # (1,)
    pairs::AbstractArray{IT, 2} # (n_pairs, 2)
    neighbour_relative_positions::AbstractArray{FT, 2} # (n_particles, kMaxNeighbourCount * 2)
end

function PerPair(n)
    n_pairs = [0] |> CT
    pairs = zeros(IT, n * kMaxNeighbourCount, 2) |> CT
    neighbour_relative_positions = zeros(FT, n * kMaxNeighbourCount, 2) |> CT
    return PerPair(n_pairs, pairs, neighbour_relative_positions)
end

@kernel function device_findneighbours!(n_particles, positions, n_pairs, pairs, neighbour_relative_positions)
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
        if r2 < kR2
            pair_index = Atomix.@atomic n_pairs[1] += 1
            pairs[pair_index, 1] = I
            pairs[pair_index, 2] = J
            neighbour_relative_positions[pair_index, 1] = dx
            neighbour_relative_positions[pair_index, 2] = dy
        end
        J += 1
    end
end

@inline function host_findneighbours!(particles::Particles, per_pair::PerPair)
    KernelAbstractions.fill!(per_pair.n_pairs, 0)
    device_findneighbours!(Backend, kThreadNumber)(
        particles.n_particles,
        particles.positions,
        per_pair.n_pairs,
        per_pair.pairs,
        per_pair.neighbour_relative_positions,
        ndrange = (kParticleNumber,),
    )
    KernelAbstractions.synchronize(Backend)
end

@kernel function device_action!(tags, accumulated_positions, pairs, neighbour_relative_positions)
    pair_I = KernelAbstractions.@index(Global)
    @inbounds I = pairs[pair_I, 1]
    @inbounds J = pairs[pair_I, 2]
    @inbounds @match (tags[I], tags[J]) begin
        (1, 1) => f11!(
            I,
            J,
            neighbour_relative_positions[pair_I, 1],
            neighbour_relative_positions[pair_I, 2],
            accumulated_positions,
        )
        (1, 2) => f12!(
            I,
            J,
            neighbour_relative_positions[pair_I, 1],
            neighbour_relative_positions[pair_I, 2],
            accumulated_positions,
        )
        (1, 3) => f13!(
            I,
            J,
            neighbour_relative_positions[pair_I, 1],
            neighbour_relative_positions[pair_I, 2],
            accumulated_positions,
        )
        (2, 1) => f21!(
            I,
            J,
            neighbour_relative_positions[pair_I, 1],
            neighbour_relative_positions[pair_I, 2],
            accumulated_positions,
        )
        (2, 2) => f22!(
            I,
            J,
            neighbour_relative_positions[pair_I, 1],
            neighbour_relative_positions[pair_I, 2],
            accumulated_positions,
        )
        (2, 3) => f23!(
            I,
            J,
            neighbour_relative_positions[pair_I, 1],
            neighbour_relative_positions[pair_I, 2],
            accumulated_positions,
        )
        (3, 1) => f31!(
            I,
            J,
            neighbour_relative_positions[pair_I, 1],
            neighbour_relative_positions[pair_I, 2],
            accumulated_positions,
        )
        (3, 2) => f32!(
            I,
            J,
            neighbour_relative_positions[pair_I, 1],
            neighbour_relative_positions[pair_I, 2],
            accumulated_positions,
        )
        (3, 3) => f33!(
            I,
            J,
            neighbour_relative_positions[pair_I, 1],
            neighbour_relative_positions[pair_I, 2],
            accumulated_positions,
        )
        (_, _) => nothing
    end
end

@inline function host_action!(particles::Particles, per_pair::PerPair)
    device_action!(Backend, kThreadNumber)(
        particles.tags,
        particles.accumulated_positions,
        per_pair.pairs,
        per_pair.neighbour_relative_positions,
        ndrange = (Array(per_pair.n_pairs)[1],),
    )
    KernelAbstractions.synchronize(Backend)
end

function main()
    particles = Particles(kParticleNumber)
    per_pair = PerPair(kParticleNumber)
    @info "warm up"
    host_findneighbours!(particles, per_pair)
    host_action!(particles, per_pair)
    @info "start"
    @time begin
        for _ in 1:kOuterLoop
            host_findneighbours!(particles, per_pair)
            host_action!(particles, per_pair)
        end
    end
    println(per_pair.n_pairs)
end

main()
