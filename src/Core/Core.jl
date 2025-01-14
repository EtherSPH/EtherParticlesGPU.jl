#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 14:52:54
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

module Core

using KernelAbstractions

include("Dimension.jl")
export AbstractDimension
export Dimension1D, Dimension2D, Dimension3D
export dimension

include("Parallel.jl")
export AbstractParallel
export Parallel
export synchronize
export toDevice, toHost

include("Domain/AbstractDomain.jl")
export AbstractDomain
export Domain2D
export indexCartesianToLinear, indexLinearToCartesian
export inside
export indexCartesianFromPosition, indexLinearFromPosition

include("Parameter.jl")
export AbstractParameter
export AbstractParameter2D, AbstractParameter3D

end
