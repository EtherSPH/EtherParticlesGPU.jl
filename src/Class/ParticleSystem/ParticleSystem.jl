#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/22 01:05:23
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

include("NamedIndex.jl")
include("ParticleSystemBase.jl")

@inline function defaultCapacityExpand(n_particles::IT)::IT where {IT <: Integer}
    return n_particles
end

abstract type AbstractParticleSystem{IT <: Integer, FT <: AbstractFloat} end

@inline function get_n_particles(
    particle_system::AbstractParticleSystem{IT, FT},
)::IT where {IT <: Integer, FT <: AbstractFloat}
    @inbounds return particle_system.host_base_.n_particles_[1]
end

@inline function get_n_capacity(
    particle_system::AbstractParticleSystem{IT, FT},
)::IT where {IT <: Integer, FT <: AbstractFloat}
    @inbounds return length(particle_system.host_base_.is_alive_)
end

@inline function toDevice!(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    particle_system::AbstractParticleSystem{IT, FT},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    transfer!(parallel, particle_system.device_base_, particle_system.host_base_)
    return nothing
end

@inline function toHost!(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    particle_system::AbstractParticleSystem{IT, FT},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    transfer!(parallel, particle_system.host_base_, particle_system.device_base_)
    return nothing
end

struct ParticleSystem{IT <: Integer, FT <: AbstractFloat} <: AbstractParticleSystem{IT, FT}
    device_base_::ParticleSystemBase{IT, FT}
    host_base_::ParticleSystemBase{IT, FT} # when parallel is CPU, a doubled memory is needed
    named_index_::NamedIndex{IT}
    parameters_named_tuple_::NamedTuple
    device_named_tuple_::NamedTuple # a combination of named_index_ and parameters_named_tuple_
end

@inline function ParticleSystem(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    n_particles::Integer,
    int_named_tuple::NamedTuple,
    float_named_tuple::NamedTuple,
    parameters_named_tuple::NamedTuple;
    capacityExpaned::Function = defaultCapacityExpand,
)::ParticleSystem{IT, FT} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    n_particles = parallel(n_particles)
    n_capacity = capacityExpaned(n_particles)
    named_index = NamedIndex{IT}(int_named_tuple, float_named_tuple)
    parameters_named_tuple = parallel(parameters_named_tuple)
    device_named_tuple = merge(get_index_named_tuple(named_index), parameters_named_tuple)
    device_base = ParticleSystemBase(parallel, named_index, n_capacity)
    cpu_parallel = Environment.Parallel{IT, FT, Array, KernelAbstractions.CPU()}()
    host_base = ParticleSystemBase(cpu_parallel, named_index, n_capacity)
    host_base.n_particles_[1] = n_particles
    host_base.is_alive_[1:n_particles] .= IT(1)
    particle_system =
        ParticleSystem{IT, FT}(device_base, host_base, named_index, parameters_named_tuple, device_named_tuple)
    toDevice!(parallel, particle_system)
    return particle_system
end
