#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/19 15:32:29
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # ! 0.041712 seconds (510 allocations: 26.250 KiB)
        # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
        # ! naive `oneAPI.jl` is even slower than `KernelAbstractions.jl`
 =#

include("../../oneapi_head.jl")

using Random

const n = 4096
const n_threads = 256
const kInnerLoop = 1000
const kOuterLoop = 10

a = randn(FT, n) |> CT
b = randn(FT, n) |> CT
c = zeros(FT, n) |> CT
ordered_index = Vector{Int}(1:n)
disordered_index = shuffle(ordered_index) |> CT
ordered_index = ordered_index |> CT

function device_vadd!(a, b, c, index)
    I = oneAPI.get_global_id()
    STRIDE = oneAPI.get_global_size()
    while I <= n
        @inbounds J = index[I]
        step = 1
        while step <= kInnerLoop
            @inbounds c[I] += a[J] + b[J]
            step += 1
        end
        I += STRIDE
    end
    return
end

function host_vadd!(a, b, c, index)
    @oneapi items = n_threads device_vadd!(a, b, c, index)
    oneAPI.synchronize()
end

@info "warm up"
host_vadd!(a, b, c, ordered_index)

@time begin
    for i in 1:kOuterLoop
        host_vadd!(a, b, c, disordered_index)
    end
end
