#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/23 03:34:23
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "NeighbourSystem" begin
    include("NeighbourSystemBaseTest.jl")
    include("ActivePairTest.jl")
    include("PeriodicBoundaryTest.jl")
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
        active_pair = [1 => 1, 1 => 2, 2 => 1]
        periodic_boundary_policy = EtherParticlesGPU.Class.NonePeriodicBoundaryPolicy
        neighbour_system =
            EtherParticlesGPU.Class.NeighbourSystem(parallel, domain, active_pair, periodic_boundary_policy)
        EtherParticlesGPU.Class.clean!(neighbour_system)
        # * test for base
        @test EtherParticlesGPU.Environment.toHost(parallel, neighbour_system.base_.contained_particle_index_count_) ==
              zeros(IT, EtherParticlesGPU.Environment.get_n(domain))
        @test EtherParticlesGPU.Environment.toHost(parallel, neighbour_system.base_.neighbour_cell_index_count_) ==
              [4, 6, 6, 4, 6, 9, 9, 6, 4, 6, 6, 4]
        @test size(neighbour_system.base_.neighbour_cell_index_count_) == (12,)
        @test size(neighbour_system.base_.neighbour_cell_index_list_) == (12, 9)
        # * test for active pair
        @test neighbour_system.active_pair_.pair_vector_ == [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1)]
        @test EtherParticlesGPU.Environment.toHost(parallel, neighbour_system.active_pair_.adjacency_matrix_) == IT[
            1 1
            1 0
        ]
        @test size(neighbour_system.periodic_boundary_.neighbour_cell_relative_position_list_) == (1, 1, 1)
    end
end
