import cvxpy as cp
import numpy as np


def OLS(returns, factRet, lambda_, K):
    """
    % Use this function to perform an OLS regression. Note that you will
    % not use lambda or K in this model (lambda is for LASSO, and K is for
    % BSS).
    """

    # *************** WRITE YOUR CODE HERE ***************
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

    # Run OLS for each asset
    for i, asset in enumerate(assets):
        y, *_ = (data[asset].values, )
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha[i] = coeffs[0]
        B[i, :]  = coeffs[1:]
        resid     = y - X.dot(coeffs)
        eps_var[i] = np.var(resid, ddof=1)

    # Compute expected returns: 
    f_mean = F.mean(axis=0)             # (8,)
   

    # Build covariance: 
    Sigma_f = np.cov(F, rowvar=False, ddof=1)     # (8,8)
    

    print(B)
    print(alpha)
    mu =  alpha + B.dot(f_mean)         # n x 1 vector of asset exp. returns
    Q  = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)         # n x n asset covariance matrix
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

    mu, Q = OLS(returns, factRet, lambda_, K)
    print("Expected Returns:", mu)
    print("Covariance Matrix:", Q)