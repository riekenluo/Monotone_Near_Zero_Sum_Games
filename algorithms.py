import time
import numpy as np
from numpy.linalg import norm

from utils import project_to_simplex_2norm


def extra_gradient(A, B, mu_1, mu_2, x_0, y_0, gamma=0.2, tolerance=1e-3, step_scale=1.0, max_iterations=1000000, stopping_criterion="distance", checking_freq=1000):
    """
    Finds the Nash equilibrium of a two-player non-zero-sum game given the payoff matrices A, B and regularizers mu_1, mu_2 using extragradient:
    u1 = - mu_1 / 2 * |x - x_0| ^ 2 + <Ax, y>
    u2 = <Bx, y> - mu_2 / 2 * |y - y_0| ^ 2

    Args:
        A, B: 2D numpy arrays representing the payoff matrices
        mu_1, mu_2: regularizers

    Returns:
        Iterates: zs 
        Gradient cnts: gradient_cnts
        Runtimes: runtimes
    """

    start_time = time.process_time()
    
    if A.shape != B.shape:
        raise ValueError("Payoff matrices must have the same shape.")
    m, n = A.shape

    L = max(norm(A.data, ord=2) + mu_1, norm(B.data, ord=2) + mu_2)
    mu = min(mu_1, mu_2) / 2
    eta = 1 / 2**0.5 / L
    eta = eta * step_scale

    # Initialize strategies (probability distributions)
    x = project_to_simplex_2norm(np.array(x_0))  # Player 1's strategy
    y = project_to_simplex_2norm(np.array(y_0))  # Player 2's strategy
    
    zs = []
    gradient_cnts = []
    runtimes = []
    loop_range = range(max_iterations)

    # Extragradient iteration
    for k in loop_range:
        # Compute gradients
        grad_x = mu_1 * (x - x_0) - A.T @ y # negative gradient
        grad_y = -B @ x + mu_2 * (y - y_0) # negative gradient
        
        if k % checking_freq == 0:
            # Extrapolation step
            x_mid = project_to_simplex_2norm(x - gamma * grad_x)
            y_mid = project_to_simplex_2norm(y - gamma * grad_y)

            # Compute gradients at the midpoint
            grad_x_mid = mu_1 * (x_mid - x_0) - A.T @ y_mid
            grad_y_mid = -B @ x_mid + mu_2 * (y_mid - y_0)
            
            current_time = time.process_time()
            runtimes.append(current_time - start_time)

            # Check for convergence
            if stopping_criterion == "distance":
                zs.append(np.concatenate((x, y)))
                gradient_cnts.append(k * 2 + k * 2 // checking_freq)
                x_plus = project_to_simplex_2norm(x - gamma * grad_x_mid)
                y_plus = project_to_simplex_2norm(y - gamma * grad_y_mid)
                x_diff = norm(x_plus - x, ord=2)
                y_diff = norm(y_plus - y, ord=2)
                if x_diff ** 2 + y_diff ** 2 < tolerance / (4/mu**2/gamma**2 - 2/mu/gamma + 16):
                    return zs, gradient_cnts, runtimes
            elif stopping_criterion == "gradient":
                zs.append(np.concatenate((x_mid, y_mid)))
                gradient_cnts.append(k * 2 + k * 2 // checking_freq)
                x_diff = np.dot(grad_x_mid, x_mid) - min(grad_x_mid)
                y_diff = np.dot(grad_y_mid, y_mid) - min(grad_y_mid)
                if x_diff + y_diff < tolerance:
                    return zs, gradient_cnts, runtimes

        # Extrapolation step
        x_mid = project_to_simplex_2norm(x - eta * grad_x)
        y_mid = project_to_simplex_2norm(y - eta * grad_y)

        # Compute gradients at the midpoint
        grad_x_mid = mu_1 * (x_mid - x_0) - A.T @ y_mid
        grad_y_mid = -B @ x_mid + mu_2 * (y_mid - y_0)

        # Update strategies
        x = project_to_simplex_2norm(x - eta * grad_x_mid)
        y = project_to_simplex_2norm(y - eta * grad_y_mid)

    # Return the equilibrium strategies
    return zs, gradient_cnts, runtimes


def optimistic_gradient(A, B, mu_1, mu_2, x_0, y_0, gamma=0.2, tolerance=1e-3, step_scale=1.0, max_iterations=1000000, alpha=1.0, stopping_criterion="distance", checking_freq=1000):
    """
    Finds the Nash equilibrium of a two-player non-zero-sum game given the payoff matrices A, B and regularizers mu_1, mu_2 using extragradient:
    u1 = - mu_1 / 2 * |x - x_0| ^ 2 + <Ax, y>
    u2 = <Bx, y> - mu_2 / 2 * |y - y_0| ^ 2

    Args:
        A, B: 2D numpy arrays representing the payoff matrices
        mu_1, mu_2: regularizers

    Returns:
        Iterates: zs 
        Gradient cnts: gradient_cnts
        Runtimes: runtimes
    """

    start_time = time.process_time()
    
    if A.shape != B.shape:
        raise ValueError("Payoff matrices must have the same shape.")
    m, n = A.shape
    
    L = max(norm(A.data, ord=2) + mu_1, norm(B.data, ord=2) + mu_2)
    mu = min(mu_1, mu_2) / 2
    eta = 1 / 2 / L
    eta = eta * step_scale

    # Initialize strategies (probability distributions)
    x = project_to_simplex_2norm(np.array(x_0))  # Player 1's strategy
    y = project_to_simplex_2norm(np.array(y_0))  # Player 2's strategy

    # Initialize previous gradients
    grad_x_prev = np.zeros_like(x)
    grad_y_prev = np.zeros_like(y)

    zs = []
    gradient_cnts = []
    runtimes = []
    loop_range = range(max_iterations)

    # Optimistic gradient iteration
    for k in loop_range:
        # Compute gradients
        grad_x = mu_1 * (x - x_0) - A.T @ y # negative gradient
        grad_y = -B @ x + mu_2 * (y - y_0) # negatie gradient
        
        if k % checking_freq == 0:
            # Extrapolation step
            x_mid = project_to_simplex_2norm(x - gamma * grad_x)
            y_mid = project_to_simplex_2norm(y - gamma * grad_y)

            # Compute gradients at the midpoint
            grad_x_mid = mu_1 * (x_mid - x_0) - A.T @ y_mid
            grad_y_mid = -B @ x_mid + mu_2 * (y_mid - y_0)
            
            current_time = time.process_time()
            runtimes.append(current_time - start_time)

            # Check for convergence
            if stopping_criterion == "distance":
                zs.append(np.concatenate((x, y)))
                gradient_cnts.append(k + k * 2 // checking_freq)
                x_plus = project_to_simplex_2norm(x - gamma * grad_x_mid)
                y_plus = project_to_simplex_2norm(y - gamma * grad_y_mid)
                x_diff = norm(x_plus - x, ord=2)
                y_diff = norm(y_plus - y, ord=2)
                if x_diff ** 2 + y_diff ** 2 < tolerance / (4/mu**2/gamma**2 - 2/mu/gamma + 16):
                    return zs, gradient_cnts, runtimes
            elif stopping_criterion == "gradient":
                zs.append(np.concatenate((x_mid, y_mid)))
                gradient_cnts.append(k + k * 2 // checking_freq)
                x_diff = np.dot(grad_x_mid, x_mid) - min(grad_x_mid)
                y_diff = np.dot(grad_y_mid, y_mid) - min(grad_y_mid)
                if x_diff + y_diff < tolerance:
                    return zs, gradient_cnts, runtimes

        # Update step using previous gradients
        x_new = project_to_simplex_2norm(x - eta * ((1+alpha) * grad_x - alpha * grad_x_prev))
        y_new = project_to_simplex_2norm(y - eta * ((1+alpha) * grad_y - alpha * grad_y_prev))

        # Update strategies and previous gradients
        x = x_new
        y = y_new
        grad_x_prev = grad_x
        grad_y_prev = grad_y

    # Return the equilibrium strategies
    return zs, gradient_cnts, runtimes


def lifted_primal_dual_minimax(A, mu_1, mu_2, x_0, y_0, x_hat, y_hat, gamma=0.2, tolerance=1e-3, delta=0.0, max_iterations=1000000, stopping_criterion="gradient", checking_freq=100, norm_A=None):
    """
    Finds the Nash equilibrium of a two-player ZERO-SUM game given the payoff matrices A, -A and regularizers mu_1, mu_2 using catalyst:
    u1 = - mu_1 / 2 * |x - x_0| ^ 2 + <Ax, y>
    u2 = - <Ax, y> - mu_2 / 2 * |y - y_0| ^ 2

    Args:
        A: 2D numpy arrays representing the payoff matrix
        mu_1, mu_2: regularizers

    Returns:
        Nash equilibrium: (x, y)
        Total gradients: total_gradients
    """
    
    if norm_A is None: 
        norm_A = norm(A.data, ord=2)

    L = norm_A + max(mu_1, mu_2)
    mu = min(mu_1, mu_2)
    kappa = norm_A / (mu_1 * mu_2) ** 0.5
    eta_x = 1 / (mu_1 * (2 * kappa))
    eta_y = 1 / (mu_2 * (2 * kappa))
    theta = 1 / (1 + 1 / (2 * kappa))

    x = project_to_simplex_2norm(np.array(x_0))
    y = project_to_simplex_2norm(np.array(y_0))
    x_prev = np.array(x)
    y_prev = np.array(y)

    zs = []
    gradient_cnts = []
    loop_range = range(max_iterations)

    for k in loop_range:
        if k % checking_freq == 0:
            # Compute gradients
            grad_x = mu_1 * (x - x_0) - A.T @ y # negative gradient
            grad_y = A @ x + mu_2 * (y - y_0) # negative gradient

            # Extrapolation step
            x_mid = project_to_simplex_2norm(x - gamma * grad_x)
            y_mid = project_to_simplex_2norm(y - gamma * grad_y)

            # Compute gradients at the midpoint
            grad_x_mid = mu_1 * (x_mid - x_0) - A.T @ y_mid
            grad_y_mid = A @ x_mid + mu_2 * (y_mid - y_0)

            # Check for convergence
            if stopping_criterion == "distance":
                zs.append(np.concatenate((x, y)))
                gradient_cnts.append(k * 2 + k * 2 // checking_freq)
                x_plus = project_to_simplex_2norm(x - gamma * grad_x_mid)
                y_plus = project_to_simplex_2norm(y - gamma * grad_y_mid)
                x_diff = norm(x_plus - x, ord=2)
                y_diff = norm(y_plus - y, ord=2)
                if x_diff ** 2 + y_diff ** 2 < tolerance / (4/mu**2/gamma**2 - 2/mu/gamma + 16):
                    return (x, y), (k + k * 2 // checking_freq)
            elif stopping_criterion == "gradient":
                zs.append(np.concatenate((x_mid, y_mid)))
                gradient_cnts.append(k * 2 + k * 2 // checking_freq)
                x_diff = np.dot(grad_x_mid, x_mid) - min(grad_x_mid)
                y_diff = np.dot(grad_y_mid, y_mid) - min(grad_y_mid)
                if x_diff + y_diff < tolerance + delta / 2 * (norm(x_mid - x_hat, ord=2) ** 2 + norm(y_mid - y_hat) ** 2):
                    return (x_mid, y_mid), (k + k * 2 // checking_freq)

        # Extrapolation
        tilde_x = x + theta * (x - x_prev)
        tilde_y = y + theta * (y - y_prev)
        x_next = project_to_simplex_2norm((x + eta_x * mu_1 * x_0 + eta_x * A.T @ tilde_y) / (1 + eta_x * mu_1))
        y_next = project_to_simplex_2norm((y + eta_y * mu_2 * y_0 - eta_y * A @ tilde_x) / (1 + eta_y * mu_2))

        x_prev, x = x, x_next
        y_prev, y = y, y_next

    return (x, y), max_iterations + max_iterations * 2 // checking_freq


def iterative_coupling_linearization(A, B, mu_1, mu_2, x_0, y_0, gamma=0.2, tolerance=1e-3, step_scale=1.0, max_iterations=1000, stopping_criterion="distance", subproblem_checking_freq=100):
    """
    Finds the Nash equilibrium of a two-player non-zero-sum game given the payoff matrices A, B and regularizers mu_1, mu_2 using extragradient:
    u1 = - mu_1 / 2 * |x - x_0| ^ 2 + <Ax, y>
    u2 = <Bx, y> - mu_2 / 2 * |y - y_0| ^ 2

    Args:
        A, B: 2D numpy arrays representing the payoff matrices
        mu_1, mu_2: regularizers

    Returns:
        Iterates: zs 
        Gradient cnts: gradient_cnts
        Runtimes: runtimes
    """
    
    start_time = time.process_time()
    
    beta = norm((0.5*(A+B)).data, ord=2)
    if 2*beta <= mu_1 and 2*beta <= mu_2:
        beta_1 = beta
        beta_2 = beta
    elif mu_1 <= 2*beta and 2*beta <= mu_2:
        beta_1 = mu_1 / 2
        beta_2 = 2 * beta ** 2 / mu_1
    else:
        beta_1 = 2 * beta ** 2 / mu_2
        beta_2 = mu_2 / 2

    mu = min(mu_1, mu_2) / 2
    norm_h = norm((0.5*(A-B)).data, ord=2)
    L = mu_1 + norm_h + mu_2
    delta = beta_1 + beta + beta_2
    if delta > 0:
        eta = min(1 / delta, 1 / mu)
    else:
        eta = 1 / mu
    eta = eta * step_scale
    err = mu * tolerance / (4 + 4 * mu * eta)
    mu_1_prox = mu_1 - beta_1 + 1 / eta
    mu_2_prox = mu_2 - beta_2 + 1 / eta
    
    x = project_to_simplex_2norm(np.array(x_0))
    y = project_to_simplex_2norm(np.array(y_0))

    total_gradients = 0
    zs = []
    gradient_cnts = []
    runtimes = []
    loop_range = range(max_iterations)

    for k in loop_range:
        # Compute gradients
        grad_x = mu_1 * (x - x_0) - A.T @ y # negative gradient
        grad_y = -B @ x + mu_2 * (y - y_0) # negative gradient

        # Extrapolation step
        x_mid = project_to_simplex_2norm(x - gamma * grad_x)
        y_mid = project_to_simplex_2norm(y - gamma * grad_y)

        # Compute gradients at the midpoint
        grad_x_mid = mu_1 * (x_mid - x_0) - A.T @ y_mid
        grad_y_mid = -B @ x_mid + mu_2 * (y_mid - y_0)
        
        current_time = time.process_time()
        runtimes.append(current_time - start_time)

        # Check for convergence
        if stopping_criterion == "distance":
            zs.append(np.concatenate((x, y)))
            gradient_cnts.append(total_gradients + k * 2)
            x_plus = project_to_simplex_2norm(x - gamma * grad_x_mid)
            y_plus = project_to_simplex_2norm(y - gamma * grad_y_mid)
            x_diff = norm(x_plus - x, ord=2)
            y_diff = norm(y_plus - y, ord=2)
            if x_diff ** 2 + y_diff ** 2 < tolerance / (4/mu**2/gamma**2 - 2/mu/gamma + 16):
                return zs, gradient_cnts, runtimes

        x_prox = ((mu_1 - beta_1) * x_0 + 1 / eta * x - beta_1 * (x - x_0) + 0.5 * (A.T + B.T) @ y) / mu_1_prox
        y_prox = ((mu_2 - beta_2) * y_0 + 1 / eta * y - beta_2 * (y - y_0) + 0.5 * (A + B) @ x) / mu_2_prox

        (x, y), gradient_cnt = lifted_primal_dual_minimax(0.5 * (A - B), mu_1_prox, mu_2_prox, x_prox, y_prox, x, y, max_iterations=100000, tolerance=err, delta=0.0, stopping_criterion="gradient", norm_A=norm_h, checking_freq=subproblem_checking_freq) # The subproblem is better conditioned, thus checked more frequently
        total_gradients += gradient_cnt 
        
    return zs, gradient_cnts, runtimes

