# syevd-hip

Benchmarks and kernels for symmetric eigensolvers and tridiagonalization on AMD GPUs.

## Layout
- `rocsolver_vs_magma/` – Compares rocSOLVER `syevd` vs MAGMA `ssyevd_gpu`.
- `sytrd/` – HIP implementation of Householder tridiagonalization (SYTRD) and a Python/LAPACK reference.

## Prerequisites
- ROCm with HIP toolchain (`hipcc`) and libraries: rocBLAS, rocSOLVER.
- MAGMA with HIP support (for `rocsolver_vs_magma`).
- Update Makefiles with your install prefixes and GPU arch (e.g., `--offload-arch=gfx942`).

## Run
- `rocsolver_vs_magma/syevd_benchmark -n <sizes...> [-v]`
  - `-n` matrix sizes (e.g., `-n 512 1024 2048`)
  - `-v` validate residuals/eigenvalues
- `sytrd/hip_sytrd -n <sizes...> [-p float|double] [-v] [-i ITERS] [-w WARMUP]`
