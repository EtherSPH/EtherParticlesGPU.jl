#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 14:52:54
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

struct Domain2D{IT <: Integer, FT <: AbstractFloat} <: AbstractDomain{IT, FT, Dimension2D}
    gap_::FT
    gap_square_::FT
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
    gap = FT(gap)
    gap_square = gap * gap
    first_x = FT(first_x)
    first_y = FT(first_y)
    last_x = FT(last_x)
    last_y = FT(last_y)
    span_x = last_x - first_x
    span_y = last_y - first_y
    n_x = floor(IT, span_x / gap)
    n_y = floor(IT, span_y / gap)
    n = n_x * n_y
    gap_x = span_x / n_x
    gap_y = span_y / n_y
    gap_x_inv = 1 / gap_x
    gap_y_inv = 1 / gap_y
    return Domain2D{IT, FT}(
        gap,
        gap_square,
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

@inline function Domain2D{IT, FT}()::Domain2D{IT, FT} where {IT <: Integer, FT <: AbstractFloat}
    return Domain2D{IT, FT}(1, 0, 0, 1, 1)
end

function Base.show(io::IO, domain::Domain2D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
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

@inline function indexCartesianToLinear(
    domain::AbstractDomain{IT, FT, Dimension2D},
    i::IT,
    j::IT,
)::IT where {IT <: Integer, FT <: AbstractFloat}
    return i + get_n_x(domain) * (j - IT(1))
end

@inline function indexLinearToCartesian(
    domain::AbstractDomain{IT, FT, Dimension2D},
    index::IT,
)::Tuple{IT, IT} where {IT <: Integer, FT <: AbstractFloat}
    n_x = get_n_x(domain)
    i = mod1(index, n_x)
    j = cld(index, n_x)
    return i, j
end

@inline function inside(
    domain::AbstractDomain{IT, FT, Dimension2D},
    x::FT,
    y::FT,
)::Bool where {IT <: Integer, FT <: AbstractFloat}
    return (get_first_x(domain) <= x <= get_last_x(domain) && get_first_y(domain) <= y <= get_last_y(domain))
end

@inline function indexCartesianFromPosition(
    domain::AbstractDomain{IT, FT, Dimension2D},
    x::FT,
    y::FT,
)::Tuple{IT, IT} where {IT <: Integer, FT <: AbstractFloat}
    # why here is `unsafe_trunc`?
    # see [link](https://github.com/JuliaGPU/oneAPI.jl/issues/441)
    # this problem quite annoys me during the whole 2025 year's Spring Festival
    # luckily, I found the solution in the issue
    i::IT = min(get_n_x(domain), device_floor(IT, (x - get_first_x(domain)) * get_gap_x_inv(domain)) + 1)
    i = max(IT(1), i)
    j::IT = min(get_n_y(domain), device_floor(IT, (y - get_first_y(domain)) * get_gap_y_inv(domain)) + 1)
    j = max(IT(1), j)
    return i, j
end

@inline function indexLinearFromPosition(
    domain::AbstractDomain{IT, FT, Dimension2D},
    x::FT,
    y::FT,
)::IT where {IT <: Integer, FT <: AbstractFloat}
    i, j = indexCartesianFromPosition(domain, x, y)
    return indexCartesianToLinear(domain, i, j)
end
