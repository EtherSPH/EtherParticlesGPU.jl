#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/24 19:38:29
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractParticleSystemBase{IT <: Integer, FT <: AbstractFloat} end

@inline function transfer!(
    ::AbstractParallel{IT, FT, CT, Backend},
    destination_base::AbstractParticleSystemBase{IT, FT},
    source_base::AbstractParticleSystemBase{IT, FT},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    KernelAbstractions.copyto!(Backend, destination_base.n_particles_, source_base.n_particles_)
    KernelAbstractions.copyto!(Backend, destination_base.is_alive_, source_base.is_alive_)
    KernelAbstractions.copyto!(Backend, destination_base.cell_index_, source_base.cell_index_)
    KernelAbstractions.copyto!(Backend, destination_base.int_properties_, source_base.int_properties_)
    KernelAbstractions.copyto!(Backend, destination_base.float_properties_, source_base.float_properties_)
    KernelAbstractions.synchronize(Backend)
    return nothing
end

struct ParticleSystemBase{IT <: Integer, FT <: AbstractFloat} <: AbstractParticleSystemBase{IT, FT}
    n_particles_::AbstractArray{IT, 1} # (1, ) on device, n_capacity_ >= n_particles_[1]
    is_alive_::AbstractArray{IT, 1} # (n_capacity, ) on device, 0: dead, 1: alive
    cell_index_::AbstractArray{IT, 1} # (n_capacity, ) on device, 0 waits for initialization
    int_properties_::AbstractArray{IT, 2} # (n_capacity, n_int_capacity) on device
    float_properties_::AbstractArray{FT, 2} # (n_capacity, n_float_capacity) on device
end

@inline function ParticleSystemBase(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    named_index::NamedIndex{IT},
    n_capacity::Integer,
)::ParticleSystemBase{IT, FT} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    n_capacity = parallel(n_capacity)
    n_particles = parallel(IT[0])
    is_alive = parallel(zeros(IT, n_capacity))
    cell_index = parallel(zeros(IT, n_capacity))
    int_properties = parallel(zeros(IT, n_capacity, get_int_n_capacity(named_index)))
    float_properties = parallel(zeros(FT, n_capacity, get_float_n_capacity(named_index)))
    return ParticleSystemBase{IT, FT}(n_particles, is_alive, cell_index, int_properties, float_properties)
end
