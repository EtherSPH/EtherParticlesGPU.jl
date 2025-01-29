#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/29 02:20:29
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # ! this works on gpu when using `CUDA.jl` as backend
 =#

include("../../cuda_head.jl")

@kernel function device_round(int32_x, float32_y)
    I = @index(Global)
    int32_x[I] = eltype(int32_x)(round(float32_y[I]))
end

a = rand(Int32, 10) |> CT
b = rand(Float32, 10) |> CT
device_round(Backend, 10)(a, b, ndrange = (10,))
KernelAbstractions.synchronize(Backend)
