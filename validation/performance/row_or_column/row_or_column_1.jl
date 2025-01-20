#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/20 15:55:38
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # * row-wise or column-wise operation on GPU is almost the same as the computation load is low
        # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
        [ Info: row add
        0.012835 seconds (4.50 k allocations: 189.062 KiB)
        [ Info: column add
        0.019637 seconds (4.50 k allocations: 189.062 KiB)
 =#

include("../../oneapi_head.jl")

const n_threads = 256
const n = 10_0000
const n_loops = 100

@kernel function device_row_add!(A)
    I = @index(Global)
    @inbounds A[I, 1] += A[I, 2]
end

@kernel function device_column_add!(A)
    I = @index(Global)
    @inbounds A[1, I] += A[2, I]
end

function host_row_add!(A)
    device_row_add!(Backend, n_threads)(A, ndrange = (size(A, 1),))
    return KernelAbstractions.synchronize(Backend)
end

function host_column_add!(A)
    device_column_add!(Backend, n_threads)(A, ndrange = (size(A, 2),))
    return KernelAbstractions.synchronize(Backend)
end

cpu_A = rand(FT, n, 2)
row_A = CT(cpu_A)
column_A = CT(cpu_A')

# warm up
@time host_row_add!(row_A)
@time host_column_add!(column_A)

@info "row add"
@time begin
    for i in 1:n_loops
        host_row_add!(row_A)
    end
end

@info "column add"
@time begin
    for i in 1:n_loops
        host_column_add!(column_A)
    end
end
