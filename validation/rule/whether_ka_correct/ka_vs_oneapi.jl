#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/19 18:59:15
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # ! KA is a little faster than oneAPI... almost the same.
            # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
            [ Info: first ka
            0.015947 seconds (720 allocations: 37.812 KiB)
            [ Info: warm up
            [ Info: second oneAPI
            0.016234 seconds (510 allocations: 26.250 KiB)
 =#

include("../../oneapi_head.jl")

using Random

const n = 1024
const n_threads = 256
const kInnerLoop = 1000
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

function device_oneapi_vadd!(a, b, c, index)
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

function host_oneapi_vadd!(a, b, c, index)
    @oneapi items = n_threads device_oneapi_vadd!(a, b, c, index)
    oneAPI.synchronize()
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
host_oneapi_vadd!(a, b, c, ordered_index)

@info "second oneAPI"
@time begin
    for i in 1:kOuterLoop
        host_oneapi_vadd!(a, b, c, disordered_index)
    end
end

oneapi_c = Array(c)

mask = ka_c .≈ oneapi_c

@assert all(mask) == true
