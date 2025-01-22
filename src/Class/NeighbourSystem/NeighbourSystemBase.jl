#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/22 16:40:48
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractNeighbourSystemBase{IT <: Integer} end

struct NeighbourSystemBase{IT <: Integer} <: AbstractNeighbourSystemBase{IT}
    contained_particle_index_count_::AbstractArray{IT, 1} # (n_cells, )
    contained_particle_index_list_::AbstractArray{IT, 2} # (n_cells, n_neighbours)
    # ! including the cell itself, this field is the only field need `atomic operation`
    neighbour_cell_index_count_::AbstractArray{IT, 1} # (n_cells, )
    neighbour_cell_index_list_::AbstractArray{IT, 2} # (n_cells, n_neighbours)
end

@kernel function device_initializeNeighbourSystem!(
    domain::AbstractDomain{IT, FT, Dimension2D},
    neighbour_cell_index_count,
    neighbour_cell_index_list,
) where {IT <: Integer, FT <: AbstractFloat}
    I::IT = KernelAbstractions.@index(Global)
    i::IT, j::IT = Environment.indexLinearToCartesian(domain, I)
    n_x::IT = Environment.get_n_x(domain)
    n_y::IT = Environment.get_n_y(domain)
    for di in -1:1
        ii::IT = i + di
        for dj in -1:1
            jj::IT = j + dj
            if ii >= 1 && ii <= n_x && jj >= 1 && jj <= n_y
                @inbounds neighbour_cell_index_count[I] += 1
                @inbounds neighbour_cell_index_list[I, neighbour_cell_index_count[I]] =
                    Environment.indexCartesianToLinear(domain, ii, jj)
            end
        end
    end
end

@kernel function device_initializeNeighbourSystem!(
    domain::AbstractDomain{IT, FT, Dimension3D},
    neighbour_cell_index_count,
    neighbour_cell_index_list,
) where {IT <: Integer, FT <: AbstractFloat}
    I::IT = KernelAbstractions.@index(Global)
    # TODO: add 3D support
end

@inline function host_initializeNeighbourSystem!(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_cell_index_count,
    neighbour_cell_index_list;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    device_initializeNeighbourSystem!(Backend, n_threads)(
        domain,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
        ndrange = (Environment.get_n(domain),),
    )
    Environment.synchronize(parallel)
    return nothing
end

@inline function NeighbourSystemBase(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
    n_threads::Integer = kDefaultThreadNumber,
)::NeighbourSystemBase{
    IT,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    n_cells = Environment.get_n(domain)
    contained_particle_index_count = parallel(zeros(IT, n_cells))
    contained_particle_index_list = parallel(zeros(IT, n_cells, max_neighbour_number))
    neighbour_cell_index_count = parallel(zeros(IT, n_cells))
    neighbour_cell_count = IT(neighbourCellCount(N))
    neighbour_cell_index_list = parallel(zeros(IT, n_cells, neighbour_cell_count))
    host_initializeNeighbourSystem!(
        parallel,
        domain,
        neighbour_cell_index_count,
        neighbour_cell_index_list;
        n_threads = n_threads,
    )
    return NeighbourSystemBase{IT}(
        contained_particle_index_count,
        contained_particle_index_list,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
    )
end
