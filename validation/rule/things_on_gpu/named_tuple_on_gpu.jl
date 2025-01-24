#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/23 17:43:19
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # * NamedTuple on GPU is allowed.
            # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
 =#

include("../../oneapi_head.jl")

@kernel function device_vadd!(a, x::NamedTuple)
    I = @index(Global)
    a[I] += x.a
    a[I] *= x.b
end

function host_vadd!(a, x::NamedTuple)
    device_vadd!(Backend, 2)(a, x, ndrange = (2,))
    KernelAbstractions.synchronize(Backend)
end

a = randn(FT, 2) |> CT
x = (a = 1.0f0, b = IT(2), c = 3.0f0)

@info "before:"
@info "a = $(a)"
host_vadd!(a, x)
@info "after vadd:"
@info "a = $(a)"
println(x)
