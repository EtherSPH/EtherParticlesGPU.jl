#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 17:01:15
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

using KernelAbstractions
using CUDA
using Random

const IT = Int64
const FT = Float32
const CT = CUDA.CuArray
const Backend = CUDA.CUDABackend()
