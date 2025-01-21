#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/21 16:35:32
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractParameter{IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension} end

@inline function dimension(
    ::AbstractParameter{IT, FT, Dimension},
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return IT(Environment.dimension(Dimension))
end

abstract type AbstractParameter2D{IT <: Integer, FT <: AbstractFloat} <: AbstractParameter{IT, FT, Dimension2D} end
abstract type AbstractParameter3D{IT <: Integer, FT <: AbstractFloat} <: AbstractParameter{IT, FT, Dimension3D} end
