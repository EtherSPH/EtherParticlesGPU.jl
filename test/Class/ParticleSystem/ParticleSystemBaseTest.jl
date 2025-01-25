#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/26 03:58:24
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "ParticleSystemBase" begin
    @testset "oneAPI" begin
        using Pkg
        Pkg.add("oneAPI")
        using oneAPI
        IT = Int32
        FT = Float32
        CT = oneAPI.oneArray
        Backend = oneAPI.oneAPIBackend()
        parallel = EtherParticlesGPU.Parallel{IT, FT, CT, Backend}()
        dim = 2
        neighbour_count = 50
        int_named_tuple = (Tag = 1, nCount = 1, nIndex = 1 * neighbour_count)
        float_named_tuple = (
            RVec = dim,
            Mass = 1,
            Density = 1,
            Volume = 1,
            VelocityVec = dim,
            dVelocityVec = dim,
            dDensity = 1,
            Pressure = 1,
            StrainMat = dim * dim,
            dStrainMat = dim * dim,
            StressMat = dim * dim,
            nRVec = dim * neighbour_count,
            nR = neighbour_count,
            nW = neighbour_count,
            nDW = neighbour_count,
        )
        named_index = EtherParticlesGPU.Class.NamedIndex{IT}(int_named_tuple, float_named_tuple)
        n_capacity = 100
        particle_system_base = EtherParticlesGPU.Class.ParticleSystemBase(parallel, named_index, n_capacity)
        @test size(particle_system_base.n_particles_) == (1,)
        @test size(particle_system_base.is_alive_) == (n_capacity,)
        @test size(particle_system_base.cell_index_) == (n_capacity,)
        @test size(particle_system_base.int_properties_) == (n_capacity, 1 + 1 + 1 * neighbour_count)
        @test size(particle_system_base.float_properties_) == (
            n_capacity,
            dim +
            1 +
            1 +
            1 +
            dim +
            dim +
            1 +
            1 +
            dim * dim +
            dim * dim +
            dim * dim +
            dim * neighbour_count +
            neighbour_count +
            neighbour_count +
            neighbour_count,
        )
    end
end
