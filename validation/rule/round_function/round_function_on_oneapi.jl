#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/29 02:20:29
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # ! no, not working
 =#

include("../../oneapi_head.jl")

function device_round(int32_x, float32_y)
    I = get_global_id()
    round_float32_y::Float32 = round(float32_y[I])
    round_int32_y::Int32 = Int32(round_float32_y)
    int32_x[I] = round_int32_y
    return
end

a = rand(Int32, 10) |> CT
b = rand(Float32, 10) |> CT
@oneapi items = 10 device_round(a, b)
oneAPI.synchronize()
