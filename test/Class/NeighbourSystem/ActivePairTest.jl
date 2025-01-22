#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/23 03:55:26
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "ActivePair" begin
    @testset "oneAPI" begin
        using Pkg
        Pkg.add("oneAPI")
        using oneAPI
        IT = Int32
        FT = Float32
        CT = oneAPI.oneArray
        Backend = oneAPI.oneAPIBackend()
        parallel = EtherParticlesGPU.Parallel{IT, FT, CT, Backend}()
        pair_vector = [1 => 1, 1 => 2, 2 => 1, 1 => 3, 3 => 1]
        active_pair = EtherParticlesGPU.Class.ActivePair(parallel, pair_vector)
        @test active_pair.pair_vector_ ==
              [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1), IT(1) => IT(3), IT(3) => IT(1)]
        @test EtherParticlesGPU.toHost(parallel, active_pair.adjacency_matrix_) == IT[
            1 1 1
            1 0 0
            1 0 0
        ]
    end
end
