#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/15 14:55:32
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

const kMaxNeighbourCount = 100
const kNX = 20
const kNY = 20
const kN = kNX * kNY
const kGap = 0.1
const kR = 0.3
const kR2 = kR * kR
const kTag =3

function get_particle(I)
    i = mod1(I, kNX)
    j = cld(I, kNX)
    x = i * kGap
    y = j * kGap
    tag = rand(1:kTag) |> Int
    return x, y, tag
end
