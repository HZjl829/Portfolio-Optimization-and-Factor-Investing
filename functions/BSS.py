import cvxpy as cp
import numpy as np
import gurobipy as gp


def BSS(returns, factRet, lambda_, K):
    """
    % Use this function for the BSS model. Note that you will not use
    % lambda in this model (lambda is for LASSO).
    %
    % You should use an optimizer to solve this problem. Be sure to comment
    % on your code to (briefly) explain your procedure.

    Inputs:
      - returns : pd.DataFrame (T × N) of each asset's excess monthly returns.
      - factRet : pd.DataFrame (T × 8) with factor columns
                  ['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev'].
      - lambda_  : not used here (for LASSO).
      - K        : int, maximum number of nonzero coefficients.
    Outputs:
      - mu : np.ndarray (N,) of expected returns from the factor model.
      - Q  : np.ndarray (N, N) covariance matrix from the factor model.
    """

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------
    # Align returns & factors by date, drop missing
    data = returns.join(factRet, how='inner').dropna()

    # Build regression matrix X = [1, factors]
    factor_cols = ['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev']
    F = data[factor_cols].values  # (T, 8)
    T, p = F.shape
    X = np.hstack([np.ones((T, 1)), F])  # (T, p+1)

    assets = returns.columns
    N = len(assets)

    # storage for outputs
    alpha = np.zeros(N)
    B     = np.zeros((N, p))
    eps_var = np.zeros(N)

    # Solve a mixed-integer QP for each asset
    for i, asset in enumerate(assets):
        y = data[asset].values  # (T,)

        # Preliminary OLS to set a big-M bound
        ols_coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        M = np.max(np.abs(ols_coeffs)) * 2 + 1e-6

        # Build Gurobi model
        model = gp.Model()
        model.Params.OutputFlag = 0  # silent

        # continuous betas β₀...βₚ and binary selectors z₀...zₚ
        beta = model.addVars(p+1, lb=-M, ub=M, name="beta")
        z    = model.addVars(p+1, vtype=gp.GRB.BINARY, name="z")

        # Cardinality constraint: sum z_j ≤ K
        model.addConstr(gp.quicksum(z[j] for j in range(p+1)) <= K)

        # Big-M linking: β_j = 0 when z_j = 0
        for j in range(p+1):
            model.addConstr(beta[j] <=  M * z[j])
            model.addConstr(beta[j] >= -M * z[j])

        # 3e) Build quadratic objective: minimize ||y - Xβ||²
        Qmat = X.T @ X  # (p+1, p+1)
        cvec = -2 * (X.T @ y)  # (p+1,)
        obj = gp.QuadExpr()
        # β^T Q β term + linear term
        for j in range(p+1):
            obj.add(beta[j] * cvec[j])
            for k in range(p+1):
                obj.add(beta[j] * beta[k] * Qmat[j, k])

        model.setObjective(obj, gp.GRB.MINIMIZE)
        model.optimize()

        # 3f) Extract solution
        sol = np.array([beta[j].X for j in range(p+1)])
        alpha[i]   = sol[0]
        B[i, :]    = sol[1:]
        resid      = y - X.dot(sol)
        eps_var[i] = np.var(resid, ddof=1)

    # Compute μ and Σ as in a factor model
    f_mean = F.mean(axis=0)                   # (8,)
    Sigma_f = np.cov(F, rowvar=False, ddof=1)  # (8,8)
    

    print(B)
    mu = alpha + B.dot(f_mean)
    Q  = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)
    # ----------------------------------------------------------------------

    return mu, Q


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    returns = pd.DataFrame({
        'Asset1': [0.01, 0.02, 0.015],
        'Asset2': [0.02, 0.025, 0.03]
    })

    factRet = pd.DataFrame({
        'Mkt_RF': [0.01, 0.015, 0.02],
        'SMB': [0.005, 0.007, 0.009],
        'HML': [0.002, 0.003, 0.004],
        'RMW': [0.001, 0.002, 0.003],
        'CMA': [0.0015, 0.0025, 0.0035],
        'Mom': [0.0005, 0.0007, 0.0009],
        'ST_Rev': [0.0002, 0.0003, 0.0004],
        'LT_Rev': [0.0003, 0.0004, 0.0005]
    })

    lambda_ = 1
    K = 2

    mu, Q = BSS(returns, factRet, lambda_, K)
    print("Expected Returns:", mu)
    print("Covariance Matrix:", Q)