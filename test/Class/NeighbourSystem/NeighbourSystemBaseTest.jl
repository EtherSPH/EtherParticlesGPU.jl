#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/23 03:36:01
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "NeighbourSystemBase" begin
    @testset "oneAPI" begin
        using Pkg
        Pkg.add("oneAPI")
        using oneAPI
        IT = Int32
        FT = Float32
        CT = oneAPI.oneArray
        Backend = oneAPI.oneAPIBackend()
        parallel = EtherParticlesGPU.Parallel{IT, FT, CT, Backend}()
        domain = EtherParticlesGPU.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
        # 5 * 4 = 20 cells
        # 4 | 6 | 6 | 6 | 4
        # --|---|---|---|--
        # 6 | 9 | 9 | 9 | 6
        # --|---|---|---|--
        # 6 | 9 | 9 | 9 | 6
        # --|---|---|---|--
        # 4 | 6 | 6 | 6 | 4
        neighbour_system_base = EtherParticlesGPU.Class.NeighbourSystemBase(parallel, domain)
        @test EtherParticlesGPU.toHost(parallel, neighbour_system_base.neighbour_cell_index_count_) ==
              [4, 6, 6, 6, 4, 6, 9, 9, 9, 6, 6, 9, 9, 9, 6, 4, 6, 6, 6, 4]
        @test size(neighbour_system_base.neighbour_cell_index_count_) == (20,)
        @test size(neighbour_system_base.neighbour_cell_index_list_) == (20, 9)
    end
end
