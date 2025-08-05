CXX = hipcc
ROCBLAS_DIR = /opt/rocm

CFLAGS = -O3 -Wall -Wno-unused-result -I${ROCBLAS_DIR}/include --offload-arch=gfx942
LDFLAGS = -L${ROCBLAS_DIR}/lib -lrocblas

unblocked_householder: unblocked_householder.cpp
		$(CXX) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
		rm -f unblocked_householder