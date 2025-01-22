#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 15:51:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@inline function neighbourCellCount(dim::Intger)::typeof(dim)
    return 3^dim # 9 for 2D, 27 for 3D
end

include("NeighbourSystemBase.jl")

# Dimension infomation is not necessary for neighbour system, so i just remove it.
# each neighbour system should have a field called `base_` to store the basic properties.
abstract type AbstractNeighbourSystem{IT <: Integer, FT <: AbstractFloat} end

@inline function clean!(
    neighbour_system::AbstractNeighbourSystem{IT, FT},
)::Nothing where {IT <: Integer, FT <: AbstractFloat}
    KernelAbstractions.fill!(neighbour_system.base_.contained_particle_index_count_, IT(0))
    return nothing
end

@kernel function device_buildNeighbourSystem!(
    domain::Domain2D{IT, FT},
    neighbour_cell_index_count,
    neighbour_cell_index_list,
) where {IT <: Integer, FT <: AbstractFloat}
    I = KernelAbstractions.@index(Global)
    i, j = Environment.indexLinearToCartesian(domain, I)
    n_x = Environment.get_n_x(domain)
    n_y = Environment.get_n_y(domain)
    for di in -1:1
        ii = i + di
        for dj in -1:1
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
        neighbour_cell_index_count_,
        neighbour_cell_index_list_,
        ndrange = (Environment.get_n(domain),),
    )
    Environment.synchronize(parallel)
    return nothing
end

@inline function host_buildNeighbourSystem!(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT};
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    host_buildNeighbourSystem!(
        parallel,
        domain,
        neighbour_system.base_.neighbour_cell_index_count_,
        neighbour_system.base_.neighbour_cell_index_list_;
        n_threads = n_threads,
    )
    return nothing
end

struct CommonNeighbourSystem{IT <: Integer, FT <: AbstractFloat} <: AbstractNeighbourSystem{IT, FT}
    base_::NeighbourSystemBase{IT}
end

@inline function CommonNeighbourSystem(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
    n_threads::Integer = kDefaultThreadNumber,
)::CommonNeighbourSystem{
    IT,
    FT,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    base::NeighbourSystemBase{IT} = NeighbourSystemBase(parallel, domain; max_neighbour_number = max_neighbour_number)
    neighbour_system = CommonNeighbourSystem{IT, FT}(base)
    host_buildNeighbourSystem!(parallel, domain, neighbour_system; n_threads = n_threads)
    return neighbour_system
end

struct PeriodicNeighbourSystem{IT <: Integer, FT <: AbstractFloat} <: AbstractNeighbourSystem{IT, FT}
    base_::NeighbourSystemBase{IT}
    neighbour_cell_relative_position_list_::AbstractArray{IT, 3} # (n_cells, n_neighbours, dimension)
end

@inline function PeriodicNeighbourSystem(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
    n_threads::Integer = kDefaultThreadNumber,
)::PeriodicNeighbourSystem{
    IT,
    FT,
    max_neighbour_cell_number,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    base::NeighbourSystemBase{IT} = NeighbourSystemBase(parallel, domain; max_neighbour_number = max_neighbour_number)
    neighbour_cell_relative_position_list = parallel(zeros(IT, Environment.get_n(domain), neighbourCellCount(N), N))
    neighbour_system = PeriodicNeighbourSystem{IT, FT}(base, neighbour_cell_relative_position_list)
    host_buildNeighbourSystem!(parallel, domain, neighbour_system; n_threads = n_threads)
    return neighbour_system
end
