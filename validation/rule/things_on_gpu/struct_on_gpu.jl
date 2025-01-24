#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/20 17:03:18
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # * immutale struct or struct's struct is allowed to be passed to GPU kernel.
            # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
 =#

@kwdef struct SonType{IT <: Integer, FT <: AbstractFloat, Dimension}
    sound_speed::FT = 20.0
    background_pressure::FT = 0.0
    gamma::IT = 7
end

@kwdef struct ParentType{IT <: Integer, FT <: AbstractFloat, Dimension}
    son::SonType{IT, FT, Dimension} = SonType{IT, FT, Dimension}()
    other_int::IT = 1
    other_float::FT = 1.0
end

include("../../oneapi_head.jl")

parent = ParentType{IT, FT, 2}()
son = SonType{IT, FT, 2}()

@kernel function struct_struct_test!(x, parent)
    x[1] = parent.son.sound_speed
    x[2] = parent.son.background_pressure
    x[3] = parent.son.gamma
    x[4] = parent.other_int
    x[5] = parent.other_float
end

@kernel function struct_test!(x, parent)
    x[1] = parent.sound_speed
    x[2] = parent.background_pressure
    x[3] = parent.gamma
end

x = zeros(FT, 5) |> CT
struct_struct_test!(Backend, 1)(x, parent, ndrange = (1,))
KernelAbstractions.synchronize(Backend)
println(x)
