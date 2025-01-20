#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/20 21:04:07
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # * struct with multiple dispatch on GPU is allowed.
            # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
            # ! however, dynamic dispatch on gpu is not allowed.
            # ! which means that the dispatch type must be determined at compile time, instead of runtime.
 =#

include("../../oneapi_head.jl")

a = zeros(FT, 2) |> CT
b = randn(FT, 2) |> CT

abstract type AbstractParameters{IT <: Integer, FT <: AbstractFloat, N} end

struct Params{IT <: Integer, FT <: AbstractFloat, N} <: AbstractParameters{IT, FT, N}
    g_x::FT
    g_y::FT
    g_z::FT
    GAMMA::IT
end

const parameters_2d = Params{IT, FT, 2}(1.0, 2.0, 0.0, 4)
const parameters_3d = Params{IT, FT, 3}(1.0, 2.0, 3.0, 4)

@inline function index_add!(I, J, a, b, parameters::AbstractParameters{IT, FT, 2}) where {IT <: Integer, FT <: AbstractFloat}
    a[I] += b[J] * 2
    return a[I]
end

@inline function index_add!(I, J, a, b, parameters::AbstractParameters{IT, FT, 3}) where {IT <: Integer, FT <: AbstractFloat}
    a[I] += b[J] * 3
    return a[I]
end

@kernel function device_index_add!(a, @Const(b), parameters::AbstractParameters{IT, FT, N}) where {IT <: Integer, FT <: AbstractFloat, N}
    I = @index(Global)
    index_add!(I, I, a, b, parameters)
end

@inline function host_index_add!(a, b, parameters::AbstractParameters{IT, FT, N}) where {IT <: Integer, FT <: AbstractFloat, N}
    device_index_add!(Backend, 2)(a, b, parameters, ndrange = (2,))
    KernelAbstractions.synchronize(Backend)
end

@info "before:"
@info "a = $(a)"
@info "b = $(b)"
host_index_add!(a, b, parameters_2d)
@info "after index_add 2D:"
@info "a = $(a)"
KernelAbstractions.fill!(a, 0)
host_index_add!(a, b, parameters_3d)
@info "after index_add 3D:"
@info "a = $(a)"
