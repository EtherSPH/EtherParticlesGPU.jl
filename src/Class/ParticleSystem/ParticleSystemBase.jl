#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/24 19:38:29
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractParticleSystemBase{IT <: Integer, FT <: AbstractFloat} end

@inline function defaultCapacityExpand(n_particles::IT)::IT where {IT <: Integer}
    return n_particles
end
struct ParticleSystemBase{IT <: Integer, FT <: AbstractFloat} <: AbstractParticleSystemBase{IT, FT}
    n_particles_::AbstractArray{IT, 1} # (1, ) on device, n_capacity_ >= n_particles_[1]
    is_alive_::AbstractArray{IT, 1} # (n_capacity, ) on device, 0: dead, 1: alive
    cell_index_::AbstractArray{IT, 1} # (n_capacity, ) on device, 0 waits for initialization
    int_fields_::AbstractArray{IT, 2} # (n_capacity, n_int_capacity) on device
    float_fields_::AbstractArray{FT, 2} # (n_capacity, n_float_capacity) on device
end
