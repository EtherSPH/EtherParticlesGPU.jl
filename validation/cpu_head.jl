#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/22 18:14:09
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

using KernelAbstractions
using Random

const IT = Int32
const FT = Float32
const CT = Array
const Backend = KernelAbstractions.CPU()
