#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/19 18:00:51
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # ! KA is a little faster than CUDA as the `kInnerLoop` is small.
        # ! it seems that KA is correct, but the performance is better
        # ! KA must have done some optimization, but I don't know what it is...
        # ! when `kInnerLoop = 1`, KA is as fast as CUDA, but when `kInnerLoop = 1000_0000`, KA is 40 times faster than CUDA.
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

@kernel function device_ka_vadd!(@Const(a), @Const(b), c, @Const(index))
    I = @index(Global)
    @inbounds J = index[I]
    step = 1
    while step <= kInnerLoop
        @inbounds c[I] += a[J] + b[J]
        step += 1
    end
end

function host_ka_vadd!(a, b, c, index)
    device_ka_vadd!(Backend, n_threads)(a, b, c, index, ndrange = (n,))
    KernelAbstractions.synchronize(Backend)
end

function device_cuda_vadd!(a, b, c, index)
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

function host_cuda_vadd!(a, b, c, index)
    @cuda threads = n_threads blocks = ceil(Int, n / n_threads) device_cuda_vadd!(a, b, c, index)
    CUDA.synchronize()
end

@info "warm up"
host_ka_vadd!(a, b, c, ordered_index)

@info "first ka"
@time begin
    for i in 1:kOuterLoop
        host_ka_vadd!(a, b, c, disordered_index)
    end
end

ka_c = Array(c)
KernelAbstractions.fill!(c, 0.0f0)

@info "warm up"
host_cuda_vadd!(a, b, c, ordered_index)

@info "second cuda"
@time begin
    for i in 1:kOuterLoop
        host_cuda_vadd!(a, b, c, disordered_index)
    end
end

cuda_c = Array(c)

mask = ka_c .≈ cuda_c

@assert all(mask) == true
