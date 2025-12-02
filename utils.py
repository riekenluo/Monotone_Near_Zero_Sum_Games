import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import norm


def project_to_simplex_2norm(v):
    """Projects a vector v onto the probability simplex (2-norm).

    Args:
      v: A numpy array representing a vector.

    Returns:
      A numpy array representing the projection of v onto the probability simplex.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    rho = np.where(u + (1 - np.cumsum(u)) / (np.arange(n) + 1) > 0)[0][-1]
    lambda_ = (1 - np.sum(u[:rho + 1])) / (rho + 1)
    return np.maximum(v + lambda_, 0)


def create_sparse_matrix(m, n, k):
    """
    Creates a sparse m x n matrix with k non-zero elements randomly chosen
    and assigned random values between -1 and 1.

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        k (int): Number of non-zero elements.

    Returns:
        tuple: A tuple containing:
            - csr_matrix: The sparse matrix in CSR format.
            - float: The 2-norm of the sparse matrix.
    """

    if k > m * n:
        raise ValueError("Number of non-zero elements (k) cannot exceed m * n.")

    # Generate random row and column indices for the non-zero elements
    row_indices = np.random.choice(m, k, replace=True) #can be replaced = False if k <=min(m,n)
    col_indices = np.random.choice(n, k, replace=True) #can be replaced = False if k <=min(m,n)

    # Generate random values between -1 and 1 for the non-zero elements
    data = np.random.uniform(-1, 1, k)

    # Create the sparse matrix in COO format first for easy construction
    coo = csr_matrix((data, (row_indices, col_indices)), shape=(m, n)) # corrected this line

    # Convert to CSR format (efficient for computation)
    csr = coo # if you make it coo, you need to return coo.tocsr() to change into csr
    # Calculate the 2-norm
    matrix_norm = norm(csr.data, ord=2)

    return csr, matrix_norm
