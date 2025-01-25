#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/26 00:17:43
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

using Accessors

@kwdef struct A
    x::Vector{Int} = [1, 2, 3]
    nt::NamedTuple = (a = 1, b = 2)
end

a = A()
@reset a.nt.a = 3
println("@reset a.nt.a = 3 works")
println(a)

function reset_a_as_4!(a::A)
    @reset a.nt.a = 4
end

reset_a_as_4!(a)
println("reset_a_as_4!(a) does not work")
println(a)
