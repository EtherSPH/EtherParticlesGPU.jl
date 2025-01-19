#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/19 15:51:14
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # ! 6.494720 seconds (290 allocations: 7.344 KiB)
        # ! on NVIDIA GeForce RTX 4090
        # ! there must be something wrong... `KernelAbstractions.jl` is much faster than `CUDA.jl`
 =#

include("../../cuda_head.jl")

using Random

const n = 4096
const n_threads = 256
const kInnerLoop = 1000_0000
const kOuterLoop = 10

a = randn(FT, n) |> CT
b = randn(FT, n) |> CT
c = zeros(FT, n) |> CT
ordered_index = Vector{Int}(1:n)
disordered_index = shuffle(ordered_index) |> CT
ordered_index = ordered_index |> CT

function device_vadd!(a, b, c, index)
    I = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if I <= n
        @inbounds J = index[I]
        step = 1
        while step <= kInnerLoop
            @inbounds c[I] += a[J] + b[J]
            step += 1
        end
    end
    return
end

function host_vadd!(a, b, c, index)
    @cuda threads = n_threads blocks = ceil(Int, n / n_threads) device_vadd!(a, b, c, index)
    CUDA.synchronize()
end

@info "warm up"
host_vadd!(a, b, c, ordered_index)

@time begin
    for i in 1:kOuterLoop
        host_vadd!(a, b, c, disordered_index)
    end
end
