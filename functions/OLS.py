import cvxpy as cp
import numpy as np


def OLS(returns, factRet, lambda_, K):
    """
    Perform an OLS regression on 8 factors.

    Returns:
      mu      : n-vector of expected returns
      Q       : n×n asset covariance matrix
      adj_R2  : n-vector of adjusted R² for each asset regression
    """
    # ----------------------------------------------------------------------
    # Align returns and factors on dates, drop any missing rows
    data = returns.join(factRet, how='inner').dropna()

    # Build the design matrix X = [1, Mkt_RF, SMB, HML, RMW, CMA, Mom, ST_Rev, LT_Rev]
    F = data[['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev']].values  # (T, 8)
    T, p = F.shape
    X = np.hstack([np.ones((T, 1)), F])  # (T, 9)

    # Prepare storage
    assets = returns.columns
    N = len(assets)
    alpha   = np.zeros(N)       # intercepts
    B       = np.zeros((N, p))  # factor loadings
    eps_var = np.zeros(N)       # idiosyncratic variances
    adj_R2  = np.zeros(N)       # adjusted R²

    # Run OLS for each asset
    for i, asset in enumerate(assets):
        y = data[asset].values                   # (T,)
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha[i] = coeffs[0]
        B[i, :]  = coeffs[1:]

        resid = y - X.dot(coeffs)
        eps_var[i] = np.var(resid, ddof=1)

        # compute R²
        SSR = np.sum(resid**2)
        SST = np.sum((y - y.mean())**2)
        R2  = 1 - SSR / SST

        # count how many predictors are actually non-zero
        p_eff = np.count_nonzero(coeffs[1:])

        # adjusted R² with effective p_eff predictors
        adj_R2[i] = 1 - (1 - R2) * (T - 1) / (T - p_eff - 1)

    # Compute expected returns
    f_mean = F.mean(axis=0)             # (8,)
    mu     = alpha + B.dot(f_mean)      # (N,)

    # Build factor-model covariance
    Sigma_f = np.cov(F, rowvar=False, ddof=1)           # (8,8)
    Q       = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)  # (N,N)
    # ----------------------------------------------------------------------

    return mu, Q, adj_R2

if __name__ == "__main__":


    import pandas as pd
    # Example usage with longer data
    returns = pd.DataFrame({
        'Asset1': [0.01, 0.02, 0.015, 0.018],
        'Asset2': [0.02, 0.025, 0.03, 0.028]
    })
    
    factRet = pd.DataFrame({

        'Mkt_RF': [0.01, 0.015, 0.02, 0.018],
        'SMB': [0.005, 0.007, 0.006, 0.008],
        'HML': [0.002, 0.003, 0.004, 0.005],
        'RMW': [0.001, 0.002, 0.003, 0.004],
        'CMA': [0.002, 0.001, 0.003, 0.002],
        'Mom': [0.003, 0.004, 0.005, 0.006],
        'ST_Rev': [0.001, 0.002, 0.001, 0.003],
        'LT_Rev': [0.002, 0.003, 0.004, 0.005]
    })
    lambda_ = 1
    K = 2
    mu, Q = OLS(returns, factRet, lambda_, K)
    print("Expected Returns (mu):", mu)
    print("Covariance Matrix (Q):", Q)
    

