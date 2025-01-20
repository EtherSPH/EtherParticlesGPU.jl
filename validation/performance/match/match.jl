#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/20 16:26:20
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # * 0.603179 seconds (730 allocations: 37.969 KiB)
            # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
 =#

include("../../oneapi_head.jl")

using Match

const n_threads = 256
const n = 1000_0000
const n_loops = 10

function f11!(I, a, b)
    return a[I] += b[I]
end

function f12!(I, a, b)
    return a[I] += b[I] * 2
end

function f21!(I, a, b)
    return a[I] += b[I]
end

function f22!(I, a, b)
    return a[I] += b[I] * 2
end

function f!(I, a, b, ta, tb)
    @inbounds @match (ta[I], tb[I]) begin
        (1, 1) => f11!(I, a, b)
        (1, 2) => f12!(I, a, b)
        (2, 1) => f21!(I, a, b)
        (2, 2) => f22!(I, a, b)
        (1, 3) => f11!(I, a, b)
        (1, 4) => f12!(I, a, b)
        (2, 3) => f21!(I, a, b)
        (2, 4) => f22!(I, a, b)
        (3, 1) => f11!(I, a, b)
        (3, 2) => f12!(I, a, b)
        (3, 3) => f21!(I, a, b)
        (3, 4) => f22!(I, a, b)
        (4, 1) => f11!(I, a, b)
        (4, 2) => f12!(I, a, b)
        (4, 3) => f21!(I, a, b)
        (4, 4) => f22!(I, a, b)
    end
end

@kernel function apply_f!(a, b, ta, tb, f!)
    I = @index(Global)
    f!(I, a, b, ta, tb)
end

function host_apply_f!(a, b, ta, tb, f!)
    apply_f!(Backend, n_threads)(a, b, ta, tb, f!, ndrange = (length(a),))
    return KernelAbstractions.synchronize(Backend)
end

a = CT(randn(FT, n))
b = CT(randn(FT, n))
ta = CT(rand(1:4, n))
tb = CT(rand(1:4, n))

@info "warm up"
host_apply_f!(a, b, ta, tb, f!)

@time begin
    for i in 1:n_loops
        host_apply_f!(a, b, ta, tb, f!)
    end
end
