# Naming Rule

Generallly rule: `prefix` + `name` + `suffix`. Without causing ambiguity, the name should be as short as possible to improve both readability and coding-efficiency.
for example, `PositionVec` is better than `XVec`, `nRVec` is better than `neighbourRelativeVector`.

`Mass` is more detailed than simply one character `M`, `Density` is more specific than simply `rho` as `Rho` may be confused with other physical quantities.

1. Usually, we use `CamelCase` to name a field.
2. prefix `n` means the field is a field related to neighbour.
3. prefix `d` means the field is a field related to derivative. for example, `dDensity` means $\partial_t\rho$, `dVelocityVec` means $\partial_t\vec{v}$.
4. when `n` and `d` are both needed, we use `nD` as prefix. for example, `nDW` means $\nabla W_n$.
5. suffix `Vec` means the field is a vector. for example, `PositionVec` means $\vec{r}$, `VelocityVec` means $\vec{v}$.
6. suffix `Mat` means the field is a matrix. for example, `StrainMat` means $\tilde{\varepsilon}$, `StressMat` means $\tilde{\sigma}$.
