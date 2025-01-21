#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 19:00:47
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

struct PeriodicNeighbourSystem{IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension} <:
       AbstractNeighbourSystem{IT, FT, Dimension}
    contained_particle_index_count_::AbstractArray{IT, 1} # (n_cells, )
    contained_particle_index_list_::AbstractArray{IT, 2} # (n_cells, n_particles)
    neighbour_cell_index_count_::AbstractArray{IT, 1} # (n_cells, )
    neighbour_cell_index_list_::AbstractArray{IT, 2} # (n_cells, n_neighbours)
    neighbour_cell_displacement_list_::AbstractArray{FT, 3} # (n_cells, n_neighbours, n_dimensions)
end

@inline function PeriodicNeighbourSystem(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
    n_threads::Integer = kDefaultThreadNumber,
)::PeriodicNeighbourSystem{
    IT,
    FT,
    Dimension,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    n_cells,
    contained_particle_index_count,
    contained_particle_index_list,
    neighbour_cell_index_count,
    neighbour_cell_index_list =
        generateBasicNeighbourSystemProperties(parallel, domain, max_neighbour_number = max_neighbour_number)
    neighbour_cell_displacement_list = parallel(zeros(FT, n_cells, max_neighbour_number, Dimension))
    host_buildNeighbourSystem!(
        parallel,
        domain,
        neighbour_cell_index_count,
        neighbour_cell_index_list;
        n_threads = n_threads,
    )
    return PeriodicNeighbourSystem{IT, FT, Dimension}(
        contained_particle_index_count,
        contained_particle_index_list,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
        neighbour_cell_displacement_list,
    )
end
