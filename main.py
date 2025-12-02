import numpy as np
import scipy
from scipy.sparse import csr_matrix
import random

import argparse

from algorithms import extra_gradient, optimistic_gradient, iterative_coupling_linearization
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your script description.')
    parser.add_argument('-n', '--p1_dim', type=int, default=10000, help='Player 1 dimension')
    parser.add_argument('-m', '--p2_dim', type=int, default=10000, help='Player 2 dimension')
    parser.add_argument('-k', '--num_elements', type=int, default=100000, help='Number of non zero elements in the payoff matrix')
    parser.add_argument('-rho', '--tax_rate', type=float, default=0.001, help='Tax rate in the matrix game')
    parser.add_argument('-seed', '--random_seed', type=int, default=111, help='Random seed')
    parser.add_argument('-mu1', '--mu_1', type=float, default=0.0001, help='Regularizer of Player 1')
    parser.add_argument('-mu2', '--mu_2', type=float, default=1.0, help='Regularizer of Player 2')
    parser.add_argument('-tol', '--tolerance', type=float, default=1e-7, help='Tolerance of error')
    parser.add_argument('-scale', '--step_scale', type=float, default=1.0)
    parser.add_argument('-alg', '--algorithm', type=str, default='ICL', help='Algorithm')

    args = parser.parse_args()
    
    n = args.p1_dim  # Number of columns
    m = args.p2_dim  # Number of rows
    k = args.num_elements  # Number of non-zero elements
    rho = args.tax_rate
    RANDOM_SEED = args.random_seed
    mu_1 = args.mu_1
    mu_2 = args.mu_2
    tolerance = args.tolerance
    step_scale = args.step_scale
    alg = args.algorithm
    
    sparse_matrix = None
    datafile = f"data/data_m{m}_n{n}_k{k}_seed{RANDOM_SEED}"
    try:
        sparse_matrix = scipy.sparse.load_npz(datafile+".npz")
    except FileNotFoundError:
        pass
    if sparse_matrix is None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        sparse_matrix_not_normalized, matrix_norm = create_sparse_matrix(m, n, k)
        sparse_matrix = sparse_matrix_not_normalized / matrix_norm
        scipy.sparse.save_npz(datafile, sparse_matrix)

    sparse_matrix_positive = csr_matrix(sparse_matrix.copy(), shape=sparse_matrix.shape)
    sparse_matrix_positive.data = np.where(sparse_matrix.data > 0, sparse_matrix.data, 0)
    sparse_matrix_positive.eliminate_zeros()
    sparse_matrix_negative = csr_matrix(sparse_matrix.copy(), shape=sparse_matrix.shape)
    sparse_matrix_negative.data = np.where(sparse_matrix.data < 0, sparse_matrix.data, 0)
    sparse_matrix_negative.eliminate_zeros()
    A = (1 - rho) * sparse_matrix_positive + sparse_matrix_negative
    B = -sparse_matrix_positive - (1 - rho) * sparse_matrix_negative

    x_0 = np.ones(n) / n
    y_0 = np.ones(m) / m

    if alg == 'OGDA':
        zs, gradient_cnts, runtimes = optimistic_gradient(A, B, mu_1=args.mu_1, mu_2=args.mu_2, x_0=x_0, y_0=y_0, tolerance=tolerance, step_scale=step_scale, stopping_criterion="distance", max_iterations=10000000, checking_freq=1000)
        print("Total gradients:", gradient_cnts[-1])
    elif alg == 'ICL':
        zs, gradient_cnts, runtimes = iterative_coupling_linearization(A, B, mu_1=args.mu_1, mu_2=args.mu_2, x_0=x_0, y_0=y_0, tolerance=tolerance, step_scale=step_scale, stopping_criterion="distance", max_iterations=10000, subproblem_checking_freq=100)
        print("Total gradients:", gradient_cnts[-1])
    elif alg == 'EG':
        zs, gradient_cnts, runtimes = extra_gradient(A, B, mu_1=args.mu_1, mu_2=args.mu_2, x_0=x_0, y_0=y_0, tolerance=tolerance, step_scale=step_scale, stopping_criterion="distance", max_iterations=10000000, checking_freq=1000)
        print("Total gradients:", gradient_cnts[-1])

    np.savez(f"logs/logs_m{m}_n{n}_k{k}_mu1{mu_1}_mu2{mu_2}_rho{rho}_seed{RANDOM_SEED}_tol{tolerance}_alg{alg}_scale{step_scale}", zs=zs, gradient_cnts=gradient_cnts, runtimes=runtimes)
    