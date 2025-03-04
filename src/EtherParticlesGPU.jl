#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 14:52:54
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
 =#

module EtherParticlesGPU

#=
  name rule:
  1. variable name: some_variable
  2. variable in struct: some_variable_
  3. function name: someFunctuion, someFunction!
  4. function to attain variable: get_some_variable_function
  5. type name: SomeType
  6. struct name: SomeStruct
  7. constant name: kSomeConstant
  8. module name: SomeModule
  9. macro name: @some_macro
  10. file name: SomeFile.jl
  11. file name not in src: some_file.jl
=#

include("Environment/Environment.jl")
using EtherParticlesGPU.Environment
include("Class/Class.jl")
using EtherParticlesGPU.Class
include("Math/Math.jl")
using EtherParticlesGPU.Math
include("Algorithm/Algorithm.jl")

include("JuliaVista/JuliaVista.jl")

end # module EtherParticlesGPU
