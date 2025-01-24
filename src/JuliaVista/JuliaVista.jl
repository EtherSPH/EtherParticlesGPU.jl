#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/24 20:02:52
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

module JuliaVista

using PyCall
# build PyCall and install PyVista

numpy = PyCall.pyimport("numpy")
pyvista = PyCall.pyimport("pyvista")
# TODO: a simple interface to `pyvista` in `Julia` which focuses on simple `vtk` data processing especially for `PolyData`.`

end
