#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/25 00:06:45
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "ParticleSystem" begin
    include("NamedIndexTest.jl")
    include("ParticleSystemBaseTest.jl")
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
        parameters_named_tuple = (sound_speed = 340.0, gravity_x = 0.0, gravity_y = -9.8, gamma = 7)
        capacity_expaned(n)::typeof(n) = n + 100
        n_particles = 100
        particle_system = EtherParticlesGPU.Class.ParticleSystem(
            parallel,
            n_particles,
            int_named_tuple,
            float_named_tuple,
            parameters_named_tuple;
            capacityExpaned = capacity_expaned,
        )
        @test EtherParticlesGPU.Class.get_n_particles(particle_system) == n_particles
        @test EtherParticlesGPU.Class.get_n_capacity(particle_system) == capacity_expaned(n_particles)
        @test Array(particle_system.device_base_.n_particles_)[1] == n_particles
        @test length(Array(particle_system.device_base_.is_alive_)) == capacity_expaned(n_particles)
        @test sum(Array(particle_system.device_base_.is_alive_)) == n_particles
        @test typeof(particle_system.device_base_.int_properties_) <: CT
        @test typeof(particle_system.device_base_.float_properties_) <: CT
        @test typeof(particle_system.host_base_.int_properties_) == Array{IT, 2}
        @test typeof(particle_system.host_base_.float_properties_) == Array{FT, 2}
        @test typeof(particle_system.device_named_tuple_.RVec) == IT
        @test typeof(particle_system.device_named_tuple_.Tag) == IT
        @test typeof(particle_system.device_named_tuple_.sound_speed) == FT
        @test typeof(particle_system.device_named_tuple_.gravity_x) == FT
        @test typeof(particle_system.device_named_tuple_.gamma) == IT
    end
end
