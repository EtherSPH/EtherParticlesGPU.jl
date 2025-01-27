#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/26 22:20:18
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

module Math

@inline function offset(start::Integer, index::Integer)::typeof(start)
    return start + index - 1
end

@inline function offset(start::Integer, index::Integer, step::Integer)::typeof(start)
    return start + (index - 1) * step
end

export offset

end
