#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 01:11:20
  @ license: MIT
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractDomain{IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}} end

@inline function dimension(
    ::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return IT(N)
end

@inline function gap(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.gap_
end

@inline function n_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.n_x_
end

@inline function n_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.n_y_
end

@inline function n(
    domain::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.n_
end

@inline function first_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.first_x_
end

@inline function last_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.last_x_
end

@inline function first_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.first_y_
end

@inline function last_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.last_y_
end

@inline function span_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.span_x_
end

@inline function span_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.span_y_
end

@inline function gap_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.gap_x_
end

@inline function gap_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.gap_y_
end

@inline function gap_x_inv(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.gap_x_inv_
end

@inline function gap_y_inv(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{N}}
    return domain.gap_y_inv_
end

include("Domain2D.jl")
# TODO: Domain3D.jl