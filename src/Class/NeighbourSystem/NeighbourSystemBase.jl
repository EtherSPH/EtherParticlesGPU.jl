#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/22 16:40:48
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

struct NeighbourSystemBase{IT <: Integer}
    contained_particle_index_count_::AbstractArray{IT, 1} # (n_cells, )
    contained_particle_index_list_::AbstractArray{IT, 2} # (n_cells, n_neighbours)
    # ! including the cell itself, this field is the only field need `atomic operation`
    neighbour_cell_index_count_::AbstractArray{IT, 1} # (n_cells, )
    neighbour_cell_index_list_::AbstractArray{IT, 2} # (n_cells, n_neighbours)
end

@inline function NeighbourSystemBase(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
)::NeighbourSystemBase{
    IT,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    n_cells = Environment.get_n(domain)
    contained_particle_index_count = parallel(zeros(IT, n_cells))
    contained_particle_index_list = parallel(zeros(IT, n_cells, max_neighbour_number))
    neighbour_cell_index_count = parallel(zeros(IT, n_cells))
    neighbour_cell_count = IT(neighbourCellCount(N))
    neighbour_cell_index_list = parallel(zeros(IT, n_cells, neighbour_cell_count))
    return NeighbourSystemBase{IT}(
        contained_particle_index_count,
        contained_particle_index_list,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
    )
end
