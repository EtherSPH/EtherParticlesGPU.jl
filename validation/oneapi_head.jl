#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 17:00:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

using KernelAbstractions
using oneAPI
using Random

const IT = Int32
const FT = Float32
const CT = oneAPI.oneArray
const Backend = oneAPI.oneAPIBackend()
