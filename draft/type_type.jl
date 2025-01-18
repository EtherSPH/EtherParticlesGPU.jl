#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 15:11:00
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

@inline function dimension(::Type{<:AbstractDimension{N}}) where {N}
    return N
end

# abstract type AbstractDomain{IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension} end

# struct Domain2D{IT <: Integer, FT <: AbstractFloat} <: AbstractDomain{IT, FT, Dimension2D} end

# const domain_2d = Domain2D{Int32, Float64}()

# @inline function dimension(
#     ::AbstractDomain{IT, FT, Dimension},
# ) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
#     return IT(dimension(Dimension))
# end

# struct K
#     domain
# end

# const k = K(domain_2d)

# @time for _ in 1:10^9
#     dimension(Dimension2D)
#     dimension(domain_2d)
# end

# * it's zero-cost abstraction
