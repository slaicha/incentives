import numpy as np
import json
from scipy.optimize import minimize
from scipy.stats import expon

np.random.seed(4) # to make the experiment reproducible

def client_utility_fn(R, c, q):
    """
    Compute the utility for each client.

    The utility function is defined as:
        U_n = R_n - c_n * q_n

    Args:
        R (array-like): Array of reward values for each client.
        c (array-like): Array of cost values for each client.
        q (array-like): Array of q values for each client.

    Returns:
        np.array: Array of utility values for each client.
    """
    return np.array(R) - np.array(c) * np.array(q)

def reward_complete_scenario(q, c):
    """
    Calculate the reward sequence R_n based on:
    R_n = c_n * q_n

    Args:
        q (array-like): Array of q values.
        c (array-like): Array of c values corresponding to each q.

    Returns:
        np.array: Array of rewards R for each n.
    """
    return np.array(c) * np.array(q)


def complete_server_utility_fn(q, gamma1, gamma2, a, G, c):
    """
    Compute the server's utility in the complete information scenario.

    The utility function is defined as:
        U_server = gamma1 * sum((a_n^2 / q_n) * G_n^2) + gamma2 * sum(c_n * q_n)

    Args:
        q (array-like): Array of q values for each client.
        gamma1 (float): Weight parameter for the loss term.
        gamma2 (float): Weight parameter for the budget term.
        a (array-like): Array of scaling factors for each client.
        G (array-like): Array of gradient norm estimates for each client.
        c (array-like): Array of cost values for each client.

    Returns:
        float: The computed server utility value.
    """
    loss = gamma1 * np.sum((a**2 / q) * (G))
    budget = gamma2 * np.sum(c * q)
    return loss + budget


def incomplete_server_utility_fn(q, gamma1, gamma2, a, G, c):
    """
    Compute the server's utility in the incomplete information scenario.

    The utility function is defined as:
        U_server = gamma1 * sum((a_n^2 / q_n) * G_n^2) 
                   + gamma2 * (N * c_1 * q_1 + sum((N-n+1) * c_n * (q_n - q_{n-1})))

    Args:
        q (array-like): Array of q values for each client.
        gamma1 (float): Weight parameter for the loss term.
        gamma2 (float): Weight parameter for the budget term.
        a (array-like): Array of scaling factors for each client.
        G (array-like): Array of gradient norm estimates for each client.
        c (array-like): Array of cost values for each client.

    Returns:
        float: The computed server utility value.
    """
    loss = gamma1 * np.sum((a**2 / q) * (G))
    budget = gamma2 * (len(q) * c[0] * q[0] + np.sum([(len(q)-n+1) * c[n] * (q[n] - q[n-1]) for n in range(1, len(q))]))
    return loss + budget



def reward_incomplete_scenario(q, c):
    """
    Calculate the reward sequence R_n based on:
    R_n = c_1 * q_1 + sum_{k=2}^{n} c_k * (q_k - q_{k-1})

    Args:
        q (array-like): Array of q values, should be decreasing.
        c (array-like): Array of c values corresponding to each q.

    Returns:
        np.array: Array of rewards R for each n.
    """
    N = len(q)
    R = np.zeros(N)  
    R[0] = c[0] * q[0]  # First reward

    for n in range(1, N):
        R[n] = R[n-1] + c[n] * (q[n] - q[n-1])

    return R


def constraint_diff(q):
    """
    This function ensures that q follows a non-decreasing order constraint by computing the difference between two consecutive constraints
    (i.e., q_n ≥ q_{n-1} for all n).

    Args:
        q (array-like): Array of q values.

    Returns:
        np.array: Differences between consecutive elements of q.
    """
    return np.diff(q)


def solve_optimization(scenario, G):
    """
    Solve the optimization problem for the given scenario.

    Args:
        scenario (str): Either "complete" for the complete information case 
                        or any other value for the incomplete case.

    Returns:
        tuple: (Optimal q values as an array, optimal objective function value)
    """
    G = np.array(G)
    with open("/home/as1233/incentives/advanced-pytorch/config_utility.json", "r") as f:
        config = json.load(f)

    # set the parameters
    N = config["N"]
    # start_value = N * 10
    # step_size = 10
    # c = np.array([start_value - i * step_size for i in range(N)])
    c = np.array(expon.rvs(scale=0.9, size=N)*100) # *0.1, 10 for 40 clients, *0.9, 100 for 10 clients
    c = np.sort(c)[::-1]
    print("Costs: ", c)
    gamma1 = config["gamma1"]
    gamma2 = config["gamma2"]
    q0 = np.sort(np.random.uniform(low=0.001, high=1.0, size=N))
    a = np.ones(N)/N
    bounds_q = [(0.001, 1) for _ in range(N)]
    scenario = config["scenario"]
    
    if scenario == "complete":
        sol = minimize(complete_server_utility_fn, q0, args=(gamma1, gamma2, a, G, c), bounds=bounds_q)
        # print("sol.x: ", sol.x)
    if scenario == "incomplete": 
        sol = minimize(incomplete_server_utility_fn, q0, args=(gamma1, gamma2, a, G, c), bounds=bounds_q,
                       constraints={'type': 'ineq', 'fun': constraint_diff})
        # print("sol.x: ", sol.x)

    return sol.x, sol.fun


# q, objective = solve_optimization("incomplete", G=config["Gn"])
# print("q: ", q)
# print("objective :", objective)