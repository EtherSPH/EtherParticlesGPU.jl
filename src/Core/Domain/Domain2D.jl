#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 01:17:34
  @ license: MIT
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

struct Domain2D{IT <: Integer, FT <: AbstractFloat} <: AbstractDomain{IT, FT, Dimension2D}
    gap_::FT
    n_x_::IT
    n_y_::IT
    n_::IT
    first_x_::FT
    last_x_::FT
    first_y_::FT
    last_y_::FT
    span_x_::FT
    span_y_::FT
    gap_x_::FT
    gap_y_::FT
    gap_x_inv_::FT
    gap_y_inv_::FT
end

function Domain2D{IT, FT}(
    gap::Real,
    first_x::Real,
    first_y::Real,
    last_x::Real,
    last_y::Real,
)::Domain2D{IT, FT} where {IT <: Integer, FT <: AbstractFloat}
    gap = convert(FT, gap)
    first_x = FT(first_x)
    first_y = FT(first_y)
    last_x = FT(last_x)
    last_y = FT(last_y)
    span_x = last_x - first_x
    span_y = last_y - first_y
    n_x = ceil(IT, span_x / gap)
    n_y = ceil(IT, span_y / gap)
    n = n_x * n_y
    gap_x = span_x / n_x
    gap_y = span_y / n_y
    gap_x_inv = 1 / gap_x
    gap_y_inv = 1 / gap_y
    return Domain2D{IT, FT}(
        gap,
        n_x,
        n_y,
        n,
        first_x,
        last_x,
        first_y,
        last_y,
        span_x,
        span_y,
        gap_x,
        gap_y,
        gap_x_inv,
        gap_y_inv,
    )
end

function Base.show(io::IO, domain::Domain2D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    println(io, "Domain2D{$IT, $FT}(")
    println(io, "    gap: ", domain.gap_)
    println(io, "    n_x: ", domain.n_x_)
    println(io, "    n_y: ", domain.n_y_)
    println(io, "    n: ", domain.n_)
    println(io, "    first_x: ", domain.first_x_)
    println(io, "    last_x: ", domain.last_x_)
    println(io, "    first_y: ", domain.first_y_)
    println(io, "    last_y: ", domain.last_y_)
    println(io, "    span_x: ", domain.span_x_)
    println(io, "    span_y: ", domain.span_y_)
    println(io, "    gap_x: ", domain.gap_x_)
    println(io, "    gap_y: ", domain.gap_y_)
    println(io, "    gap_x_inv: ", domain.gap_x_inv_)
    println(io, "    gap_y_inv: ", domain.gap_y_inv_)
    return println(io, ")")
end
