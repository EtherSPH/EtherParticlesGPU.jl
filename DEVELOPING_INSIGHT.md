# Development Insights

## Design Data Structures on CPU as Complex as Possible while Keep Them Simple on GPU

Not only in `Julia`, data-structure designing such as `template`, `class`, `struct` in other language like `cpp` is quite annoying as it comes to programming on GPU.

Complex data structures like `Dict`, `Tuple` should be avoided on GPU. Instead, use `immutable struct` and `Array` as much as possible to make it work on GPU. However, each time a kernel is launched, some preprocessing is needed on CPU host side. Thus, it is better to design a complex data structure on CPU side to make the dispatching easier and the code reuse rate higher.

When it comes to GPU, the data structure should be as simple as possible. But the `Type` assertion can be passed to the GPU kernel to attain the same effect as complex data structures, just as what `template` does in `cpp`. An [experiment](validation/rule/struct_on_gpu/type_on_gpu.jl) is carried out to test such feasibility.

## Performance & Easy-to-Code & Extensibility is an Impossible Trinity

**Performance** needs quite fixed data structures and algorithms, which is quite opposite to the **easy-to-code** and **extensibility**. The more complex the data structure is, the more difficult it is to write and maintain the code. The more flexible the data structure is, the more difficult it is to optimize the code.

API exposed to the user should be as simple as possible, but the internal implementation should be as complex as possible to make the code reusable and maintainable. The balance between **performance** and **easy-to-code** is quite difficult to achieve.

In this project, I decide to *sacrifice* the extremly high **performance** for the sake of **easy-to-code** and **extensibility**. The **performance** can be improved by optimizing the code and the data structure, but the **easy-to-code** and **extensibility** is quite difficult to achieve once the API is exposed to the user.

Generally speaking, I decide:

```txt
Performance < Easy-to-Code < Extensibility
```