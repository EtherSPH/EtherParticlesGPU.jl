# Note

1. `Function` in `Dict` will raise type instability in julia. However, it seems the type insability is almost zero-cost in this case. `Tuple` is also allowed as a substitution for `Dict` in this case.
2. !!!Warning `oneAPI` has bug when launching kernels at current version(2025.01.15). Frequent kernel launch will cause performance drop. See [link](https://www.phoronix.com/review/intel-b580-opencl-january/2) for details.
3. `if-else` is not a good choice for performance at device side. From my test, a 9 branch if-else will have lower performance when the float operation is more than 5*3 times.
4. However, although split partcles by type and apply different operations from host side, in real simulation, it maybe not the case. Function like `apply!(ps1, ps2, f!)` will run empty kernel if `ps1` or `ps2` is not neighbour. Thus, a better choice maybe apply by `pairs`.