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
    

    
    mu =  alpha + B.dot(f_mean)         # n x 1 vector of asset exp. returns
    Q  = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)         # n x n asset covariance matrix
    # ----------------------------------------------------------------------

    return mu, Q
