#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/20 16:07:34
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # * row-wise operation will be a little faster than column-wise operation on GPU
        # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
        [ Info: row add
        0.225104 seconds (4.50 k allocations: 189.062 KiB)
        [ Info: column add
        0.334020 seconds (4.50 k allocations: 189.062 KiB)
 =#

include("../../oneapi_head.jl")

const n_threads = 256
const n = 100_0000
const n_loops = 100

@kernel function device_row_op!(A)
    I = @index(Global)
    @inbounds A[I, 1] += A[I, 1] * A[I, 2] + A[I, 3] # row-wise operation
end

@kernel function device_column_op!(A)
    I = @index(Global)
    @inbounds A[1, I] += A[1, I] * A[2, I] + A[3, I] # column-wise operation
end

function host_row_op!(A)
    device_row_op!(Backend, n_threads)(A, ndrange = (size(A, 1),))
    return KernelAbstractions.synchronize(Backend)
end

function host_column_op!(A)
    device_column_op!(Backend, n_threads)(A, ndrange = (size(A, 2),))
    return KernelAbstractions.synchronize(Backend)
end

cpu_A = rand(FT, n, 3)
row_A = CT(cpu_A)
column_A = CT(cpu_A')

# warm up
@time host_row_op!(row_A)
@time host_column_op!(column_A)

@info "row op"
@time begin
    for i in 1:n_loops
        host_row_op!(row_A)
    end
end

@info "column op"
@time begin
    for i in 1:n_loops
        host_column_op!(column_A)
    end
end
