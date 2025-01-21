#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 00:57:33
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "Parallel" begin
    @testset "CPU" begin
        IT = Int32
        FT = Float32
        CT = Array
        Backend = KernelAbstractions.CPU()
        parallel = EtherParticlesGPU.Parallel{IT, FT, CT, Backend}()
        @test EtherParticlesGPU.Environment.IntType(parallel) == IT
        @test EtherParticlesGPU.Environment.FloatType(parallel) == FT
        @test EtherParticlesGPU.Environment.ContainerType(parallel) == CT
        @test EtherParticlesGPU.Environment.getBackend(parallel) == Backend
        @test parallel(UInt8(1)) == 1
        @test parallel(1.0) ≈ 1.0f0
        @test parallel(Int32.(1:3)) == [1, 2, 3]
        @test parallel(Float64.(1:3)) ≈ [1.0f0, 2.0f0, 3.0f0]
        EtherParticlesGPU.synchronize(parallel)
        @test EtherParticlesGPU.toHost(parallel, [1, 2, 3]) == [1, 2, 3]
        @test EtherParticlesGPU.toHost(parallel, [1.0f0, 2.0f0, 3.0f0]) ≈ [1.0f0, 2.0f0, 3.0f0]
    end
    @testset "oneAPI" begin
        using Pkg
        Pkg.add("oneAPI")
        using oneAPI
        IT = Int32
        FT = Float32
        CT = oneAPI.oneArray
        Backend = oneAPI.oneAPIBackend()
        parallel = EtherParticlesGPU.Parallel{IT, FT, CT, Backend}()
        @test EtherParticlesGPU.Environment.IntType(parallel) == IT
        @test EtherParticlesGPU.Environment.FloatType(parallel) == FT
        @test EtherParticlesGPU.Environment.ContainerType(parallel) == CT
        @test EtherParticlesGPU.Environment.getBackend(parallel) == Backend
        @test parallel(UInt8(1)) == 1
        @test parallel(1.0) ≈ 1.0f0
        @test parallel(Int32.(1:3)) |> Array == [1, 2, 3]
        @test parallel(Float64.(1:3)) |> Array ≈ [1.0f0, 2.0f0, 3.0f0]
        EtherParticlesGPU.synchronize(parallel)
        @test EtherParticlesGPU.toDevice(parallel, Int32.(1:3)) |> Array == [1, 2, 3]
        @test EtherParticlesGPU.toDevice(parallel, Float16.(1:3)) |> Array ≈ [1.0f0, 2.0f0, 3.0f0]
        @test EtherParticlesGPU.toHost(parallel, CT([1, 2, 3])) == [1, 2, 3]
        @test EtherParticlesGPU.toHost(parallel, CT([1.0f0, 2.0f0, 3.0f0])) ≈ [1.0f0, 2.0f0, 3.0f0]
    end
end
