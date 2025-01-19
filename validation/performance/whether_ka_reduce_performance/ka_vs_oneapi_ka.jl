#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/19 15:17:03
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # ! 0.032929 seconds (720 allocations: 37.812 KiB)
        # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
        # ! `KernelAbstractions.jl` is faster than naive `oneAPI.jl`
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

@kernel function device_vadd!(@Const(a), @Const(b), c, @Const(index))
    I = @index(Global)
    @inbounds J = index[I]
    step = 1
    while step <= kInnerLoop
        @inbounds c[I] += a[J] + b[J]
        step += 1
    end
end

function host_vadd!(a, b, c, index)
    device_vadd!(Backend, n_threads)(a, b, c, index, ndrange = (n,))
    KernelAbstractions.synchronize(Backend)
end

@info "warm up"
host_vadd!(a, b, c, ordered_index)

@time begin
    for i in 1:kOuterLoop
        host_vadd!(a, b, c, disordered_index)
    end
end
