#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/23 14:01:37
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "PeriodicBoundary" begin
    @testset "oneAPI" begin
        using Pkg
        Pkg.add("oneAPI")
        using oneAPI
        IT = Int32
        FT = Float32
        CT = oneAPI.oneArray
        Backend = oneAPI.oneAPIBackend()
        parallel = EtherParticlesGPU.Parallel{IT, FT, CT, Backend}()
        domain = EtherParticlesGPU.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.6, 0.6)
        # 4 * 3 = 12 cells
        # 4 | 6 | 4
        # --|---|--
        # 6 | 9 | 6
        # --|---|--
        # 6 | 9 | 6
        # --|---|--
        # 4 | 6 | 4
        periodic_boundary = EtherParticlesGPU.Class.PeriodicBoundary(
            parallel,
            domain,
            EtherParticlesGPU.Class.NonePeriodicBoundaryPolicy,
        )
        @test size(periodic_boundary.neighbour_cell_relative_position_list_) == (1, 1, 1)
        periodic_boundary = EtherParticlesGPU.Class.PeriodicBoundary(
            parallel,
            domain,
            EtherParticlesGPU.Class.PeriodicBoundaryPolicy2DAlongX,
        )
        @test size(periodic_boundary.neighbour_cell_relative_position_list_) == (12, 9, 2)
    end
end
