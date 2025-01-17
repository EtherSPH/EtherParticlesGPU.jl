#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 18:33:00
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

include("../../oneapi_head.jl")

const n_threads = 256
const n = 10_0000
const n_loops = 50
const n_loops_inner = 100

struct ABC
    as::Vector{CT}
end

@inline function f11!(I, x, y, z)
    for _ in 1:n_loops_inner
        z[I] += x[I] + y[I]
    end
end

@inline function f12!(I, x, y, z)
    for _ in 1:n_loops_inner
        z[I] += x[I] + y[I] * 2
    end
end

@inline function f13!(I, x, y, z)
    for _ in 1:n_loops_inner
        z[I] += x[I] + y[I] * 3
    end
end

@inline function f22!(I, x, y, z)
    for _ in 1:n_loops_inner
        z[I] += x[I] + y[I] * 4
    end
end

@inline function f23!(I, x, y, z)
    for _ in 1:n_loops_inner
        z[I] += x[I] + y[I] * 5
    end
end

@inline function f33!(I, x, y, z)
    for _ in 1:n_loops_inner
        z[I] += x[I] + y[I] * 6
    end
end

const f_dict = Dict(
    (1, 1) => f11!,
    (1, 2) => f12!,
    (1, 3) => f13!,
    (2, 1) => f12!,
    (2, 2) => f22!,
    (2, 3) => f23!,
    (3, 1) => f13!,
    (3, 2) => f23!,
    (3, 3) => f33!,
)

@kernel function device_apply_f!(x, y, z, f!)
    I = @index(Global)
    f!(I, x, y, z)
end

function host_apply_f!(a, b, c, z)
    device_apply_f!(Backend, n_threads)(a, a, z, f11!, ndrange = (length(z),))
    device_apply_f!(Backend, n_threads)(b, a, z, f12!, ndrange = (length(z),))
    device_apply_f!(Backend, n_threads)(c, a, z, f13!, ndrange = (length(z),))
    device_apply_f!(Backend, n_threads)(a, b, z, f12!, ndrange = (length(z),))
    device_apply_f!(Backend, n_threads)(b, b, z, f22!, ndrange = (length(z),))
    device_apply_f!(Backend, n_threads)(c, b, z, f23!, ndrange = (length(z),))
    device_apply_f!(Backend, n_threads)(a, c, z, f13!, ndrange = (length(z),))
    device_apply_f!(Backend, n_threads)(b, c, z, f23!, ndrange = (length(z),))
    device_apply_f!(Backend, n_threads)(c, c, z, f33!, ndrange = (length(z),))
    KernelAbstractions.synchronize(Backend)
end

a = rand(1:3, n) |> CT
b = rand(1:3, n) |> CT
c = rand(1:3, n) |> CT
z = CT(zeros(IT, n))
abc = ABC([a, b, c])

@info "warm up"
host_apply_f!(a, b, c, z)
KernelAbstractions.fill!(z, 0)

@time for _ in 1:n_loops
    host_apply_f!(a, b, c, z)
end
