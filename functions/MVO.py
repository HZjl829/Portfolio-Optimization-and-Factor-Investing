import cvxpy as cp
import numpy as np


def MVO(mu, Q, targetRet):
    """
    Construct a no‐short‐sales MVO portfolio targeting 'targetRet'.
    Inputs:
      - mu        : np.ndarray, shape (n,), expected returns (e.g. monthly)
      - Q         : np.ndarray, shape (n,n), covariance matrix
      - targetRet : float, desired portfolio return level
    Returns:
      - x.value   : np.ndarray, shape (n,), optimal weights
    """
     
    n = len(mu)
    
    # Decision variable: portfolio weights
    x = cp.Variable(n)
    
    # Objective: minimize portfolio variance x' Q x
    obj = cp.Minimize(cp.quad_form(x, Q))
    
    # Constraints:
    #    – Full investment
    #    – Achieve at least targetRet
    #    – No short sales
    constraints = [
        cp.sum(x) == 1,
        mu @ x >= targetRet,
        x >= 0
    ]
    
    # Solve the QP
    prob = cp.Problem(obj, constraints)
    prob.solve()  # you can specify solver=cp.GUROBI if available
    
    # Return the numeric weights
    return x.value

if __name__ == "__main__":
    # Example usage
    mu = np.array([0.01, 0.02, 0.015])
    Q = np.array([[0.0001, 0.00005, 0.00002],
                  [0.00005, 0.0002, 0.00003],
                  [0.00002, 0.00003, 0.00015]])
    targetRet = 0.018
    
    weights = MVO(mu, Q, targetRet)
    print("Optimal Weights:", weights)
    print("Sum of Weights:", np.sum(weights))  # Should be close to 1
    print("Target Return Achieved:", mu @ weights)  # Should be >= targetRet
    print("Portfolio Variance:", weights.T @ Q @ weights)  # Portfolio variance
    print("Portfolio Standard Deviation:", np.sqrt(weights.T @ Q @ weights))  # Portfolio standard deviation