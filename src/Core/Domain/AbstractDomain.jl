#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 14:52:54
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractDomain{IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension} end

@inline function dimension(
    ::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return IT(dimension(Dimension))
end

@inline function get_gap(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.gap_
end

@inline function get_n_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.n_x_
end

@inline function get_n_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.n_y_
end

@inline function get_n(
    domain::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.n_
end

@inline function get_first_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.first_x_
end

@inline function get_last_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.last_x_
end

@inline function get_first_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.first_y_
end

@inline function get_last_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.last_y_
end

@inline function get_span_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.span_x_
end

@inline function get_span_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.span_y_
end

@inline function get_gap_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.gap_x_
end

@inline function get_gap_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.gap_y_
end

@inline function get_gap_x_inv(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.gap_x_inv_
end

@inline function get_gap_y_inv(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return domain.gap_y_inv_
end

function Base.show(io::IO, domain::AbstractDomain{IT, FT, Dimension2D}) where {IT <: Integer, FT <: AbstractFloat}
    println(io, "Domain2D{$IT, $FT}(")
    println(io, "    gap: ", get_gap(domain))
    println(io, "    n_x: ", get_n_x(domain))
    println(io, "    n_y: ", get_n_y(domain))
    println(io, "    n: ", get_n(domain))
    println(io, "    first_x: ", get_first_x(domain))
    println(io, "    last_x: ", get_last_x(domain))
    println(io, "    first_y: ", get_first_y(domain))
    println(io, "    last_y: ", get_last_y(domain))
    println(io, "    span_x: ", get_span_x(domain))
    println(io, "    span_y: ", get_span_y(domain))
    println(io, "    gap_x: ", get_gap_x(domain))
    println(io, "    gap_y: ", get_gap_y(domain))
    println(io, "    gap_x_inv: ", get_gap_x_inv(domain))
    println(io, "    gap_y_inv: ", get_gap_y_inv(domain))
    return println(io, ")")
end

include("Domain2D.jl")
# TODO: Domain3D.jl
