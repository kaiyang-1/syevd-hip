# Makefile for Householder tridiagonalization

# Compiler and flags
CXX = hipcc
ROCBLAS_DIR = /opt/rocm

CFLAGS = -O3 -Wall -Wno-unused-result -I${ROCBLAS_DIR}/include --offload-arch=gfx942
LDFLAGS = -L${ROCBLAS_DIR}/lib -lrocblas -lrocsolver

# Targets
TARGET = hip_sytrd

# Source files
SRC = hip_sytrd.cpp

# Default target
all: $(TARGET)

# Build
$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

# Clean
clean:
	rm -f $(TARGET)

# Test
test: $(TARGET)
	./$(TARGET) -n 128 256 512 -v

.PHONY: all clean test
