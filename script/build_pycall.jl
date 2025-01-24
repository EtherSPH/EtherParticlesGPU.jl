#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/25 01:04:13
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

using Pkg
Pkg.add("Conda")
Pkg.add("PyCall")
ENV["PYTHON"] = ""
Pkg.build("PyCall")
