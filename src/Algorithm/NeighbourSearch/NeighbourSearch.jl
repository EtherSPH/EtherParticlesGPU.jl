#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/26 17:05:05
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@kernel function device_insertParticlesToCells!(
    domain_2d::AbstractDomain{IT, FT, Dimension2D},
    ps_is_alive,
    ps_cell_index,
    @Const(ps_float_properties),
    ns_contained_particle_index_count,
    ns_contained_particle_index_list,
    index_position_vector::IT,
) where {IT <: Integer, FT <: AbstractFloat}
    I::IT = KernelAbstractions.@index(Global)
    @inbounds if ps_is_alive[I] == 1
        @inbounds x::FT = ps_float_properties[I, index_position_vector]
        @inbounds y::FT = ps_float_properties[I, index_position_vector + 1]
        if Environment.inside(domain_2d, x, y)
            # cell_index::IT = Environment.indexLinearFromPosition(domain_2d, x, y)
            cell_index::IT = 1
            @inbounds ps_cell_index[I] = cell_index
            particle_in_cell_index::IT = Atomix.@atomic ns_contained_particle_index_count[cell_index] += IT(1)
            @inbounds ns_contained_particle_index_list[cell_index, particle_in_cell_index] = I
        else
            @inbounds ps_is_alive[I] = IT(0)
            @inbounds ps_cell_index[I] = IT(0)
        end
    end
end

# TODO: add 3D support

@inline function host_insertParticlesToCells!(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    particle_system::AbstractParticleSystem{IT, FT},
    neighbour_system::AbstractNeighbourSystem{IT, FT, PeriodicBoundaryPolicy},
    index_position_vector::IT,
    n_particles::Integer;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    device_insertParticlesToCells!(Backend, n_threads)(
        domain,
        particle_system.device_base_.is_alive_,
        particle_system.device_base_.cell_index_,
        particle_system.device_base_.float_properties_,
        neighbour_system.base_.contained_particle_index_count_,
        neighbour_system.base_.contained_particle_index_list_,
        index_position_vector,
        ndrange = (n_particles,),
    )
    Environment.synchronize(parallel)
    return nothing
end

@inline function search!(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    particle_system::AbstractParticleSystem{IT, FT},
    neighbour_system::AbstractNeighbourSystem{IT, FT, PeriodicBoundaryPolicy};
    n_threads::Integer = kDefaultThreadNumber,
    position_vector_symbol::Symbol = :PositionVec,
    tag_symbol::Symbol = :Tag,
    neighbour_count_symbol::Symbol = :nCount,
    neighbour_index_symbol::Symbol = :nIndex,
    neighbour_relative_vector_symbol::Symbol = :nRVec,
    neighbour_distance_symbol::Symbol = :nR,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    clean!(neighbour_system)
    n_particles::IT = Class.get_n_particles(particle_system)
    index_position_vector::IT = getfield(particle_system.device_named_tuple_, position_vector_symbol)
    host_insertParticlesToCells!(
        parallel,
        domain,
        particle_system,
        neighbour_system,
        index_position_vector,
        n_particles;
        n_threads = n_threads,
    )
    # index_tag::IT = getfield(particle_system.device_named_tuple_, tag_symbol)
    # index_neighbour_count::IT = getfield(particle_system.device_named_tuple_, neighbour_count_symbol)
    # index_neighbour_relative_vector::IT =
    #     getfield(particle_system.device_named_tuple_, neighbour_relative_vector_symbol)
    # index_neighbour_distance::IT = getfield(particle_system.device_named_tuple_, neighbour_distance_symbol)
    return nothing
end
