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

        # Trailing submatrix size
        trailing_size = n - start_col

        # Initialize U and V matrices for accumulating rank-2p updates
        U = np.zeros((trailing_size, p))
        V = np.zeros((trailing_size, p))

        # Process each column within the block
        for j in range(start_col, end_col):
            # Extract the column to be updated
            a_col = A[j+1:, j].copy()

            # Block column index
            j_idx = j - start_col

            # Apply previously accumulated updates
            if j > start_col:
                U_prev = U[j_idx+1:, :j_idx]
                V_prev = V[j_idx+1:, :j_idx]

                a_col -= V_prev @ U[j_idx, :j_idx].T + U_prev @ V[j_idx, :j_idx].T

            # Compute the Householder vector
            u_j = np.zeros(trailing_size)
            norm_a = np.linalg.norm(a_col)

            if norm_a < 1e-10:  # Skip zero vector
                u_j[j_idx+1] = 1.0
            else:
                sign = 1.0 if a_col[0] >= 0 else -1.0
                alpha = -sign * norm_a

                # Construct the Householder vector u_j
                u_j[j_idx+1] = np.sqrt((1 - a_col[0] / alpha))

                if abs(alpha * u_j[j_idx+1]) > 1e-10:
                    u_j[j_idx+2:] = -a_col[1:] / (alpha * u_j[j_idx+1])

            # Compute y_j = (A_orig - UVᵀ - VUᵀ)u_j
            if j_idx > 0:
                U_prev = U[:, :j_idx]
                V_prev = V[:, :j_idx]

                y_j = (A[start_col:, start_col:] - U_prev @ V_prev.T - V_prev @ U_prev.T) @ u_j
            else:
                y_j = A[start_col:, start_col:] @ u_j

            # Compute v_j = y_j - (1/2)(y_jᵀu_j)u_j
            v_j = y_j - 0.5 * np.dot(y_j, u_j) * u_j

            # Store u_j and v_j
            U[:, j_idx] = u_j
            V[:, j_idx] = v_j

        # Apply symmetric rank-2p update for the block
        block_size = end_col - start_col
        U_block = U[:, :block_size]
        V_block = V[:, :block_size]

        A[start_col:, start_col:] -= U_block @ V_block.T + V_block @ U_block.T
    
    # Extract the tridiagonal part
    D = np.diag(np.diag(A))
    E = (np.diag(A, k=1) + np.diag(A, k=-1)) / 2

    return D, E

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

        u = np.zeros(n - i)
        u[0] = 0
        u[1] = np.sqrt((1 - a[0] / alpha))
        u[2:] = -a[1:] / (alpha * u[1])

        # Compute y and v
        y = A[i:, i:] @ u
        v = y - 0.5 * np.dot(y, u) * u

        # Apply symmetric rank-2 update
        A[i:, i:] = A[i:, i:] - np.outer(u, v) - np.outer(v, u)

    # Extract the tridiagonal part
    D = np.diag(np.diag(A))
    E = (np.diag(A, k=1) + np.diag(A, k=-1)) / 2

    return D, E

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
            D_blocked, E_blocked = block_householder_tridiagonalize(A_copy, block_size)
            blocked_times.append(time.time() - start)
        avg_blocked = np.mean(blocked_times)
        
        # Test non-blocked method
        non_blocked_times = []
        for _ in range(num_trials):
            A_copy = A.copy()
            start = time.time()
            D_non_blocked, E_non_blocked = householder_tridiagonalize(A_copy)
            non_blocked_times.append(time.time() - start)
        avg_non_blocked = np.mean(non_blocked_times)

        # Verify correctness
        diff = max(np.max(np.abs(D_blocked - D_non_blocked)), np.max(np.abs(E_blocked - E_non_blocked)))
        
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