#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 17:20:52
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

include("../oneapi_head.jl")

@inline function f11!(I, x, y)
    x[I] += y[I]
end

@inline function f12!(I, x, y)
    x[I] += y[I] * 2
end

@inline function f21!(I, x, y)
    x[I] += y[I] * 3
end

const f_dict = Dict(
    (1, 1) => f11!,
    (1, 2) => f12!,
    (2, 1) => f21!,
)

@kernel function device_apply_f!(x, y, f!)
    I = @index(Global)
    f!(I, x, y)
end

function host_apply_f!(abc, f_dict)
    device_apply_f!(Backend, 5)(abc[1], abc[2], f_dict[(1, 1)], ndrange = (5,))
    KernelAbstractions.synchronize(Backend)
end

a = rand(1:3, 5) |> CT
b = rand(1:3, 5) |> CT
va = [a, b]

host_apply_f!(Backend, 5)(va, f_dict)