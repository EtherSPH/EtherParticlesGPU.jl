#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 15:51:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

"""
    AbstractNeighbourSystem{IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    should bascially have fields:
        - `contained_particle_index_count_::AbstractArray{IT, 1}` (n_cells, )
        - `contained_particle_index_list_::AbstractArray{IT, 2}` (n_cells, n_particles)
        - `neighbour_cell_index_count_::AbstractArray{IT, 1}` (n_cells, )
        - `neighbour_cell_index_list_::AbstractArray{IT, 2}` (n_cells, n_neighbours)
"""
abstract type AbstractNeighbourSystem{IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension} end

@inline function clean!(
    neighbour_system::AbstractNeighbourSystem{IT, FT, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    KernelAbstractions.fill!(neighbour_system.contained_particle_index_count_, 0)
    return nothing
end

@inline function generateBasicNeighbourSystemProperties(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
)::Tuple where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    n_cells = get_n(domain)
    contained_particle_index_count = parallel(zeros(IT, n_cells))
    contained_particle_index_list = parallel(zeros(IT, n_cells, max_neighbour_number))
    neighbour_cell_index_count = parallel(zeros(IT, n_cells))
    max_neighbour_cell_number = IT(3^Dimension) # 9 for 2D, 27 for 3D
    neighbour_cell_index_list = parallel(zeros(IT, n_cells, max_neighbour_cell_number))
    return n_cells,
    contained_particle_index_count,
    contained_particle_index_list,
    neighbour_cell_index_count,
    neighbour_cell_index_list
end

@kernel function device_buildNeighbourSystem!(
    domain::Domain2D{IT, FT},
    neighbour_cell_index_count,
    neighbour_cell_index_list,
) where {IT <: Integer, FT <: AbstractFloat}
    I = KernelAbstractions.@index(Global)
    i, j = indexLinearToCartesian(I, domain.n_cells)
    n_x = get_n_x(domain)
    n_y = get_n_y(domain)
    for di in -1:1
        for dj in -1:1
            ii = i + di
            jj = j + dj
            if ii >= 1 && ii <= n_x && jj >= 1 && jj <= n_y
                @inbounds neighbour_cell_index_count[I] += 1
                @inbounds neighbour_cell_index_list[I, neighbour_cell_index_count[I]] =
                    indexCartesianToLinear(domain, ii, jj)
            end
        end
    end
end

# TODO: add 3D support

@inline function host_buildNeighbourSystem!(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_cell_index_count,
    neighbour_cell_index_list;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    device_buildNeighbourSystem!(Backend, n_threads)(
        domain,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
        ndrange = (get_n(domain),),
    )
    synchronize(parallel)
    return nothing
end

include("CommonNeighbourSystem.jl")
include("PeriodicNeighbourSystem.jl")
