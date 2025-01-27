#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 15:51:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@inline function neighbourCellCount(dim::IT)::typeof(dim) where {IT <: Integer}
    return 3^dim # 9 for 2D, 27 for 3D
end

include("NeighbourSystemBase.jl")
include("ActivePair.jl")
include("PeriodicBoundary.jl")

"""
    - `base_`: the base of neighbour system
    - `active_pair_`: the active pair of neighbour system
    - `periodic_boundary_`: the periodic boundary of neighbour system
"""
abstract type AbstractNeighbourSystem{
    IT <: Integer,
    FT <: AbstractFloat,
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
} end

@inline function get_n_cells(
    neighbour_system::AbstractNeighbourSystem{IT, FT, PeriodicBoundaryPolicy},
)::IT where {IT <: Integer, FT <: AbstractFloat, PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy}
    return length(neighbour_system.base_.contained_particle_index_count_)
end

struct NeighbourSystem{IT <: Integer, FT <: AbstractFloat, PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy} <:
       AbstractNeighbourSystem{IT, FT, PeriodicBoundaryPolicy}
    base_::NeighbourSystemBase{IT}
    active_pair_::ActivePair{IT}
    periodic_boundary_::PeriodicBoundary{FT, PeriodicBoundaryPolicy}
end

@inline function NeighbourSystem(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    active_pair::Vector{<:Pair{<:Integer, <:Integer}},
    periodic_boundary_policy::Type{<:AbstractPeriodicBoundaryPolicy};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
    n_threads::Integer = kDefaultThreadNumber,
)::NeighbourSystem{
    IT,
    FT,
    periodic_boundary_policy,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    base = NeighbourSystemBase(parallel, domain; max_neighbour_number = max_neighbour_number, n_threads = n_threads)
    active_pair = ActivePair(parallel, active_pair)
    periodic_boundary = PeriodicBoundary(parallel, domain, periodic_boundary_policy)
    return NeighbourSystem{IT, FT, periodic_boundary_policy}(base, active_pair, periodic_boundary)
end

@inline function clean!(
    neighbour_system::AbstractNeighbourSystem{IT, FT, PeriodicBoundaryPolicy},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy}
    KernelAbstractions.fill!(neighbour_system.base_.contained_particle_index_count_, IT(0))
    return nothing
end

@inline function Base.show(
    io::IO,
    neighbour_system::AbstractNeighbourSystem{IT, FT, PeriodicBoundaryPolicy},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy}
    print(io, "NeighbourSystem{$IT, $FT, $PeriodicBoundaryPolicy}\n")
    print(io, "  cells: $(get_n_cells(neighbour_system))\n")
    print(io, "  cell neighbours: $(size(neighbour_system.base_.neighbour_cell_index_list_, 2))\n")
    print(io, "  active pair: $(neighbour_system.active_pair_.pair_vector_)\n")
    print(io, "  periodic boundary policy: $PeriodicBoundaryPolicy\n")
    return nothing
end
