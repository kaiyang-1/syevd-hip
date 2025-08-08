# Makefile for Householder tridiagonalization

# Compiler and flags
CXX = hipcc
ROCBLAS_DIR = /opt/rocm

CFLAGS = -O3 -Wall -Wno-unused-result -I${ROCBLAS_DIR}/include --offload-arch=gfx942
LDFLAGS = -L${ROCBLAS_DIR}/lib -lrocblas

# Targets
TARGET = householder_benchmark

# Source files
SRC = main.cpp householder_common.cpp blocked_householder.cpp unblocked_householder.cpp

# Default target - build the unified benchmark
all: $(TARGET)

# Build unified benchmark
$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

# Clean
clean:
	rm -f $(TARGET)

# Test benchmark with different modes
test_blocked: $(TARGET_BLOCKED)
	./$(TARGET) -a blocked -n 128 256 512 -b 32 -v

test_unblocked: $(TARGET_UNBLOCKED)
	./$(TARGET) -a unblocked -n 128 256 512 -v

# Run performance comparison
compare: $(TARGET)
	@echo "=== Performance Comparison ==="
	./$(TARGET) -n 128 256 512 1024 -a both -b 32

.PHONY: all clean test_blocked test_unblocked compare
