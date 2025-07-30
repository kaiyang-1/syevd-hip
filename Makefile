CXX = hipcc
MAGMA_DIR = /path/to/magma/installation
ROCSOLVER_DIR = /path/to/rocsolver/installation
ROCBLAS_DIR = /path/to/rocblas/installation

CFLAGS = -O3 -Wall -Wno-unused-result -I${MAGMA_DIR}/include -I${ROCSOLVER_DIR}/include -I${ROCBLAS_DIR}/include --offload-arch=gfx942
LDFLAGS = -L${MAGMA_DIR}/lib -lmagma -L${ROCSOLVER_DIR}/lib -lrocsolver -L${ROCBLAS_DIR}/lib -lrocblas

syevd_benchmark: syevd_benchmark.cpp
        $(CXX) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
        rm -f syevd_benchmark