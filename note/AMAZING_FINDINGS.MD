# Amazing Findings

## `round`-like function on GPU

It's amazing that `round`-like function such as `floor`, `ceil`, `trunc`, `round` are not available on GPU when `oneAPI` is used as the backend. This is a big limitation for me since I need to use `round`-like function in my code. I hope this will be fixed soon.

I will go to the `oneAPI` forum to ask about this issue.

Luckily, `CUDA.jl` overides the `round`-like function and provides the `round`-like function on GPU. This is a good news for me. I will use `CUDA.jl` as the backend for now.

See [files](../validation/rule/round_function/round_function_on_gpu.jl) for more details.