# Makefile for Householder tridiagonalization

# Compiler and flags
CXX = hipcc
ROCBLAS_DIR = /opt/rocm

CFLAGS = -O3 -Wall -Wno-unused-result -I${ROCBLAS_DIR}/include --offload-arch=gfx942
LDFLAGS = -L${ROCBLAS_DIR}/lib -lrocblas

# Targets
TARGET_BLOCKED = blocked_householder
TARGET_UNBLOCKED = unblocked_householder

# Source files
SRC_BLOCKED = blocked_householder.cpp
SRC_UNBLOCKED = unblocked_householder.cpp

# Default target
all: $(TARGET_BLOCKED) $(TARGET_UNBLOCKED)

# Build blocked version
$(TARGET_BLOCKED): $(SRC_BLOCKED)
	$(CXX) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Build unblocked version
$(TARGET_UNBLOCKED): $(SRC_UNBLOCKED)
	$(CXX) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Clean
clean:
	rm -f $(TARGET_BLOCKED) $(TARGET_UNBLOCKED)

# Run tests
test_blocked: $(TARGET_BLOCKED)
	./$(TARGET_BLOCKED) -n 128 256 512 -b 32 -v

test_unblocked: $(TARGET_UNBLOCKED)
	./$(TARGET_UNBLOCKED) -n 128 256 512 -v

# Compare performance
compare: $(TARGET_BLOCKED) $(TARGET_UNBLOCKED)
	@echo "=== Blocked Householder Performance ==="
	./$(TARGET_BLOCKED) -n 128 256 512 1024 -b 32
	@echo ""
	@echo "=== Unblocked Householder Performance ==="
	./$(TARGET_UNBLOCKED) -n 128 256 512 1024

.PHONY: all clean test_blocked test_unblocked compare
