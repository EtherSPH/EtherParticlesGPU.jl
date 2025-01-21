#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 01:13:35
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

@testset "Domain" begin
    IT = Int32
    FT = Float32
    x_0 = -1.0
    y_0 = -2.0
    x_1 = 3.0
    y_1 = 3.0
    gap = 0.15
    domain_2d = EtherParticlesGPU.Domain2D{IT, FT}(gap, x_0, y_0, x_1, y_1)

    @test EtherParticlesGPU.Environment.dimension(domain_2d) == 2
    @test EtherParticlesGPU.Environment.get_gap(domain_2d) ≈ gap
    @test EtherParticlesGPU.Environment.get_gap_square(domain_2d) ≈ gap * gap
    @test EtherParticlesGPU.Environment.get_n_x(domain_2d) == 27
    @test EtherParticlesGPU.Environment.get_n_y(domain_2d) == 34
    @test EtherParticlesGPU.Environment.get_n(domain_2d) == 27 * 34
    @test EtherParticlesGPU.Environment.get_first_x(domain_2d) ≈ x_0
    @test EtherParticlesGPU.Environment.get_last_x(domain_2d) ≈ x_1
    @test EtherParticlesGPU.Environment.get_first_y(domain_2d) ≈ y_0
    @test EtherParticlesGPU.Environment.get_last_y(domain_2d) ≈ y_1
    @test EtherParticlesGPU.Environment.get_span_x(domain_2d) ≈ x_1 - x_0
    @test EtherParticlesGPU.Environment.get_span_y(domain_2d) ≈ y_1 - y_0
    @test EtherParticlesGPU.Environment.get_gap_x(domain_2d) ≈ (x_1 - x_0) / 27
    @test EtherParticlesGPU.Environment.get_gap_y(domain_2d) ≈ (y_1 - y_0) / 34
    @test EtherParticlesGPU.Environment.get_gap_x_inv(domain_2d) ≈
          1 / EtherParticlesGPU.Environment.get_gap_x(domain_2d)
    @test EtherParticlesGPU.Environment.get_gap_y_inv(domain_2d) ≈
          1 / EtherParticlesGPU.Environment.get_gap_y(domain_2d)

    @test EtherParticlesGPU.indexCartesianToLinear(domain_2d, IT(2), IT(3)) == 2 + (3 - 1) * 27
    @test EtherParticlesGPU.indexLinearToCartesian(domain_2d, IT(2 + (3 - 1) * 27)) == (2, 3)
    @test EtherParticlesGPU.inside(domain_2d, 1.5f0, 0.3f0) == true
    @test EtherParticlesGPU.indexCartesianFromPosition(domain_2d, 1.5f0, 0.3f0) == (17, 16)
    @test EtherParticlesGPU.indexLinearFromPosition(domain_2d, 1.5f0, 0.3f0) == 17 + (16 - 1) * 27
end
