#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/20 16:13:16
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # * row-wise operation will be much faster than column-wise operation on GPU as data span is large
        # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
        [ Info: row add
        0.959574 seconds (4.50 k allocations: 189.062 KiB)
        [ Info: column add
        2.739471 seconds (4.50 k allocations: 189.062 KiB)
 =#

include("../../oneapi_head.jl")

const n_threads = 256
const n = 100_0000
const n_loops = 100

@kernel function device_row_op!(A)
    I = @index(Global)
    @inbounds A[I, 1] += A[I, 1] * A[I, 2] + A[I, 3] # row-wise operation
    @inbounds A[I, 98] += A[I, 2] * A[I, 3] + A[I, 4] # row-wise operation
    @inbounds A[I, 99] += A[I, 3] * A[I, 4] + A[I, 5] # row-wise operation
    @inbounds A[I, 100] += A[I, 4] * A[I, 5] + A[I, 6] # row-wise operation
end

@kernel function device_column_op!(A)
    I = @index(Global)
    @inbounds A[1, I] += A[1, I] * A[2, I] + A[3, I] # column-wise operation
    @inbounds A[98, I] += A[2, I] * A[3, I] + A[4, I] # column-wise operation
    @inbounds A[99, I] += A[3, I] * A[4, I] + A[5, I] # column-wise operation
    @inbounds A[100, I] += A[4, I] * A[5, I] + A[6, I] # column-wise operation
end

function host_row_op!(A)
    device_row_op!(Backend, n_threads)(A, ndrange = (size(A, 1),))
    return KernelAbstractions.synchronize(Backend)
end

function host_column_op!(A)
    device_column_op!(Backend, n_threads)(A, ndrange = (size(A, 2),))
    return KernelAbstractions.synchronize(Backend)
end

cpu_A = rand(FT, n, 100)
row_A = CT(cpu_A)
column_A = CT(cpu_A')

# warm up
@time host_row_op!(row_A)
@time host_column_op!(column_A)

@info "row add"
@time begin
    for i in 1:n_loops
        host_row_op!(row_A)
    end
end

@info "column add"
@time begin
    for i in 1:n_loops
        host_column_op!(column_A)
    end
end
