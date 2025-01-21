#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/20 21:17:59
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description: # * not too frequent `@reset` will not affect the performance
            # ! 4s for 10^8 times on my laptop Intel(R) Core(TM) i7-8650U (8) @ 4.20 GHz
 =#

using Accessors

@kwdef struct Simulation
    t::Float64 = 0.0
    dt::Float64 = 0.01
    step::Int32 = 0
end

@inline function step!(sim::Simulation)
    @reset sim.step += 1
    @reset sim.t += sim.dt
end

@inline function step(sim::Simulation)
    @reset sim.step += 1
    @reset sim.t += sim.dt
    return sim
end

sim = Simulation()
@info "initial: $sim"
step!(sim)
@info "after step!(sim): $sim" # can't change the field of sim
sim = step(sim)
@info "after step(sim): $sim" # can change the field of sim

const N = 10^8
@info "Benchmarking step(sim) for $N times requires:"
@time begin
    for _ in 1:(10^8)
        global sim = step(sim)
    end
end
