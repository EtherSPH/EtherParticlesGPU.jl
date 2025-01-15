#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 12:48:17
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

include("../../oneapi_head.jl")

using Match

const n_threads = 256
const n = 10_0000
const n_loops = 50
const n_loops_inner = 100

@inline function f11!(I, i, j, abc, z)
    for _ in 1:n_loops_inner
        z[I] += abc[I, i] + abc[I, j] * 1
    end
end

@inline function f12!(I, i, j, abc, z)
    for _ in 1:n_loops_inner
        z[I] += abc[I, i] + abc[I, j] * 2
    end
end

@inline function f13!(I, i, j, abc, z)
    for _ in 1:n_loops_inner
        z[I] += abc[I, i] + abc[I, j] * 3
    end
end

@inline function f22!(I, i, j, abc, z)
    for _ in 1:n_loops_inner
        z[I] += abc[I, i] + abc[I, j] * 4
    end
end

@inline function f23!(I, i, j, abc, z)
    for _ in 1:n_loops_inner
        z[I] += abc[I, i] + abc[I, j] * 5
    end
end

@inline function f33!(I, i, j, abc, z)
    for _ in 1:n_loops_inner
        z[I] += abc[I, i] + abc[I, j] * 6
    end
end

@kernel function device_apply_f!(abc, z)
    I = @index(Global)
    for i in 1:3
        for j in 1:3
            @inbounds L = abc[I, i]
            @inbounds R = abc[I, j]
            @match (L, R) begin
                (1, 1) => f11!(I, L, R, abc, z)
                (1, 2) => f12!(I, L, R, abc, z)
                (1, 3) => f13!(I, L, R, abc, z)
                (2, 1) => f12!(I, L, R, abc, z)
                (2, 2) => f22!(I, L, R, abc, z)
                (2, 3) => f23!(I, L, R, abc, z)
                (3, 1) => f13!(I, L, R, abc, z)
                (3, 2) => f23!(I, L, R, abc, z)
                (3, 3) => f33!(I, L, R, abc, z)
            end
            # @match (L, R) begin
            #     (1, 1) => f11!(I, i, j, abc, z)
            #     (1, 2) => f12!(I, i, j, abc, z)
            #     (1, 3) => f13!(I, i, j, abc, z)
            #     (2, 1) => f12!(I, i, j, abc, z)
            #     (2, 2) => f22!(I, i, j, abc, z)
            #     (2, 3) => f23!(I, i, j, abc, z)
            #     (3, 1) => f13!(I, i, j, abc, z)
            #     (3, 2) => f23!(I, i, j, abc, z)
            #     (3, 3) => f33!(I, i, j, abc, z)
            # end
        end
    end
end

function host_apply_f!(abc, z)
    device_apply_f!(Backend, n_threads)(abc, z, ndrange = (length(z),))
    KernelAbstractions.synchronize(Backend)
end

abc = rand(1:3, n, 3) |> CT
z = zeros(IT, n) |> CT

@info "warm up"
host_apply_f!(abc, z)
KernelAbstractions.fill!(z, 0)

@time for _ in 1:n_loops
    host_apply_f!(abc, z)
end
