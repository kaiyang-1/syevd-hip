import numpy as np
import time

# Function to perform block Householder tridiagonalization
# A: Symmetric matrix (n x n)
# p: Block size
def block_householder_tridiagonalize(A, p):
    n = A.shape[0]
    # Calculate the number of blocks
    num_blocks = (n - 2 + p - 1) // p

    # Process each block
    for k in range(num_blocks):
        start_col = k * p  # Start column index of the current block
        end_col = min(start_col + p, n - 2)  # End column index of the current block

        # Initialize U and V matrices for accumulating rank-2p updates
        U = np.zeros((n, p))
        V = np.zeros((n, p))

        # Process each column within the block
        for j in range(start_col, end_col):
            # Extract the column to be updated
            a_col = A[j+1:, j].copy()

            # Apply previously accumulated updates
            for i in range(start_col, j):
                i_idx = i - start_col
                u_i = U[:, i_idx]
                v_i = V[:, i_idx]

                # Extract relevant components
                scalar_u = u_i[j]
                scalar_v = v_i[j]
                u_tail = u_i[j+1:]
                v_tail = v_i[j+1:]

                # Apply the update
                a_col -= scalar_v * u_tail + scalar_u * v_tail

            # Compute the Householder vector
            norm_a = np.linalg.norm(a_col)
            if norm_a < 1e-10:  # Skip zero vector
                u_j = np.zeros(n)
                u_j[j+1] = 1.0
            else:
                sign = 1.0 if a_col[0] >= 0 else -1.0
                alpha = -sign * norm_a

                # Construct the Householder vector u_j
                u_j = np.zeros(n)
                u_j[j+1] = np.sqrt((1 - a_col[0] / alpha))

                if abs(alpha * u_j[j+1]) > 1e-10:
                    u_j[j+2:] = -a_col[1:] / (alpha * u_j[j+1])

            # Block column index
            j_idx = j - start_col

            # Compute y_j = (A_orig - UVᵀ - VUᵀ)u_j
            if j_idx > 0:
                U_prev = U[:, :j_idx]
                V_prev = V[:, :j_idx]

                y_j = (A - U_prev @ V_prev.T - V_prev @ U_prev.T) @ u_j
            else:
                y_j = A @ u_j

            # Compute v_j = y_j - (1/2)(y_jᵀu_j)u_j
            v_j = y_j - 0.5 * np.dot(y_j, u_j) * u_j

            # Store u_j and v_j
            U[:, j_idx] = u_j
            V[:, j_idx] = v_j

        # Apply symmetric rank-2p update for the block
        block_size = end_col - start_col
        U_block = U[:, :block_size]
        V_block = V[:, :block_size]

        A -= U_block @ V_block.T + V_block @ U_block.T

    return A

# Function to perform non-blocked Householder tridiagonalization
def householder_tridiagonalize(A):
    n = A.shape[0]
    for i in range(n - 2):
        # Extract the column to be updated
        a = A[i+1:, i]
        norm_a = np.linalg.norm(a)
        if norm_a < 1e-10:
            continue

        # Compute the Householder vector
        sign = 1.0 if a[0] >= 0 else -1.0
        alpha = -sign * norm_a

        u = np.zeros(n)
        u[i+1] = np.sqrt((1 - a[0] / alpha))
        u[i+2:] = -a[1:] / (alpha * u[i+1])

        # Compute y and v
        y = A @ u
        v = y - 0.5 * np.dot(y, u) * u

        # Apply symmetric rank-2 update
        A = A - np.outer(u, v) - np.outer(v, u)

    return A

def generate_symmetric_matrix(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    A = np.random.rand(n, n)
    return (A + A.T) / 2

def compare_methods(matrix_sizes, block_size=2, num_trials=5, seed=42):
    results = []
    
    for n in matrix_sizes:
        print(f"Testing matrix size: {n}x{n}")
        A = generate_symmetric_matrix(n, seed)
        
        # Test blocked method
        blocked_times = []
        for _ in range(num_trials):
            A_copy = A.copy()
            start = time.time()
            T_blocked = block_householder_tridiagonalize(A_copy, block_size)
            blocked_times.append(time.time() - start)
        avg_blocked = np.mean(blocked_times)
        
        # Test non-blocked method
        non_blocked_times = []
        for _ in range(num_trials):
            A_copy = A.copy()
            start = time.time()
            T_non_blocked = householder_tridiagonalize(A_copy)
            non_blocked_times.append(time.time() - start)
        avg_non_blocked = np.mean(non_blocked_times)
        
        # Verify correctness
        diff = np.max(np.abs(T_blocked - T_non_blocked))
        
        results.append({
            'size': n,
            'blocked_time': avg_blocked,
            'non_blocked_time': avg_non_blocked,
            'speedup': avg_non_blocked / avg_blocked,
            'max_difference': diff
        })
    
    return results

# Matrix sizes to test
matrix_sizes = [10, 50, 100, 200, 300, 400]
results = compare_methods(matrix_sizes, block_size=32, seed=42)

# Print summary table
print("\nPerformance Comparison Summary:")
print("Size\tBlocked Time\tNon-Blocked Time\tSpeedup\t\tMax Difference")
for r in results:
    print(f"{r['size']}\t{r['blocked_time']:.4f}\t\t{r['non_blocked_time']:.4f}\t\t\t{r['speedup']:.2f}x\t\t{r['max_difference']:.2e}")