#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 14:52:54
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

abstract type AbstractParallel{IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend} end

const kCPUBackend = CPU()

function Base.show(
    io::IO,
    ::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    println(io, "AbstractParallel{IT, FT, CT, Backend}(")
    println(io, "    IT: ", IT)
    println(io, "    FT: ", FT)
    println(io, "    CT: ", CT)
    println(io, "    Backend: ", Backend)
    return println(io, ")")
end

struct Parallel{IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend} <:
       AbstractParallel{IT, FT, CT, Backend} end

@inline function IntType(
    ::AbstractParallel{IT, FT, CT, Backend},
)::DataType where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return IT
end

@inline function FloatType(
    ::AbstractParallel{IT, FT, CT, Backend},
)::DataType where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return FT
end

@inline function ContainerType(
    ::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return CT
end

@inline function getBackend(
    ::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return Backend
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::IntType,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, IntType <: Integer}
    return IT(x)
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::FloatType,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, FloatType <: AbstractFloat}
    return FT(x)
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::AT,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, AT <: AbstractArray{<:Integer}}
    if CT === AT
        if IT === eltype(x)
            return x
        else
            return IT.(x)
        end
    else
        return CT(IT.(x))
    end
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::AT,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, AT <: AbstractArray{<:AbstractFloat}}
    if CT === AT
        if FT === eltype(x)
            return x
        else
            return FT.(x)
        end
    else
        return CT(FT.(x))
    end
end

@inline function synchronize(
    ::AbstractParallel{IT, FT, CT, Backend},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function toDevice(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    x::Array,
)::CT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return parallel(x)
end

@inline function toDevice(
    ::AbstractParallel{IT, FT, Array, kCPUBackend},
    x::Array,
)::Array where {IT <: Integer, FT <: AbstractFloat}
    return deepcopy(x)
end

@inline function toHost(
    ::AbstractParallel{IT, FT, CT, Backend},
    x::CT,
)::Array where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return Array(x)
end

@inline function toHost(
    ::AbstractParallel{IT, FT, Array, kCPUBackend},
    x::Array,
)::Array where {IT <: Integer, FT <: AbstractFloat}
    return deepcopy(x)
end
