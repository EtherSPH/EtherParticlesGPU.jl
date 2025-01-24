#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/22 01:05:23
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

include("ParticleSystemBase.jl")
include("NamedIndex/NamedIndex.jl")

abstract type AbstractParticleSystem{IT <: Integer, FT <: AbstractFloat} end

struct ParticleSystem{IT <: Integer, FT <: AbstractFloat} <: AbstractParticleSystem{IT, FT}
    device_base_::ParticleSystemBase{IT, FT}
    host_base_::ParticleSystemBase{IT, FT}
end

@inline function toDevice!(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    ps::AbstractParticleSystem{IT, FT},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractFloat, Backend}
    KernelAbstractions.copyto!(Backend, ps.device_base_.n_particles_, ps.host_base_.n_particles_)
    KernelAbstractions.copyto!(Backend, ps.device_base_.is_alive_, ps.host_base_.is_alive_)
    KernelAbstractions.copyto!(Backend, ps.device_base_.cell_index_, ps.host_base_.cell_index_)
    KernelAbstractions.copyto!(Backend, ps.device_base_.int_fields_, ps.host_base_.int_fields_)
    KernelAbstractions.copyto!(Backend, ps.device_base_.float_fields_, ps.host_base_.float_fields_)
    Environment.synchronize(parallel)
    return nothing
end

@inline function toHost!(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    ps::AbstractParticleSystem{IT, FT},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractFloat, Backend}
    KernelAbstractions.copyto!(Backend, ps.host_base_.n_particles_, ps.device_base_.n_particles_)
    KernelAbstractions.copyto!(Backend, ps.host_base_.is_alive_, ps.device_base_.is_alive_)
    KernelAbstractions.copyto!(Backend, ps.host_base_.cell_index_, ps.device_base_.cell_index_)
    KernelAbstractions.copyto!(Backend, ps.host_base_.int_fields_, ps.device_base_.int_fields_)
    KernelAbstractions.copyto!(Backend, ps.host_base_.float_fields_, ps.device_base_.float_fields_)
    Environment.synchronize(parallel)
    return nothing
end
