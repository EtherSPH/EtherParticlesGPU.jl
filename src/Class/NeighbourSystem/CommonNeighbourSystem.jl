#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 19:00:20
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

struct CommonNeighbourSystem{IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension} <:
       AbstractNeighbourSystem{IT, FT, Dimension}
    contained_particle_index_count_::AbstractArray{IT, 1} # (n_cells, )
    contained_particle_index_list_::AbstractArray{IT, 2} # (n_cells, n_neighbours)
    neighbour_cell_index_count_::AbstractArray{IT, 1} # (n_cells, )
    neighbour_cell_index_list_::AbstractArray{IT, 2} # (n_cells, n_neighbours)
end

@inline function CommonNeighbourSystem(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
    n_threads::Integer = kDefaultThreadNumber,
)::CommonNeighbourSystem{
    IT,
    FT,
    Dimension,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    _,
    contained_particle_index_count,
    contained_particle_index_list,
    neighbour_cell_index_count,
    neighbour_cell_index_list =
        generateBasicNeighbourSystemProperties(parallel, domain, max_neighbour_number = max_neighbour_number)
    host_buildNeighbourSystem!(
        parallel,
        domain,
        neighbour_cell_index_count,
        neighbour_cell_index_list;
        n_threads = n_threads,
    )
    return CommonNeighbourSystem{IT, FT, Dimension}(
        contained_particle_index_count,
        contained_particle_index_list,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
    )
end
