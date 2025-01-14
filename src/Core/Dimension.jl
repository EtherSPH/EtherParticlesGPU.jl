#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 00:54:10
  @ license: MIT
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

const kDimension1D = Dimension1D()
const kDimension2D = Dimension2D()
const kDimension3D = Dimension3D()

@inline function dimension(::AbstractDimension{N}) where {N}
    return N
end
