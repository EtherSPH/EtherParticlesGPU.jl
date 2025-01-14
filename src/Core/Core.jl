#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 00:53:42
  @ license: MIT
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

module Core

using KernelAbstractions

include("Dimension.jl")
export Dimension1D, Dimension2D, Dimension3D
export kDimension1D, kDimension2D, kDimension3D
export dimension

include("Parallel.jl")
export StandardParallel
export synchronize

include("Domain/AbstractDomain.jl")

end
