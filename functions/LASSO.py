import numpy as np
from sklearn.linear_model import Lasso

def LASSO(returns, factRet, lambda_, K):
    """
    Use this function for the LASSO model. Note that you will not use K
    in this model (K is for BSS).

    Returns:
      mu      : n-vector of expected returns
      Q       : n×n asset covariance matrix
      adj_R2  : n-vector of adjusted R² for each asset regression
    """
    # ----------------------------------------------------------------------
    # Align on dates & drop missing
    data = returns.join(factRet, how='inner').dropna()

    # Factor matrix F (T×8) and compute its mean/covariance
    factor_cols = ['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev']
    F = data[factor_cols].values
    T, p = F.shape
    f_mean  = F.mean(axis=0)
    Sigma_f = np.cov(F, rowvar=False, ddof=1)

    assets = returns.columns
    N = len(assets)

    # Storage
    alpha   = np.zeros(N)
    B       = np.zeros((N, p))
    eps_var = np.zeros(N)
    adj_R2  = np.zeros(N)

    # Fit a Lasso for each asset
    for i, asset in enumerate(assets):
        y = data[asset].values
        model = Lasso(alpha=lambda_, fit_intercept=True, max_iter=10000)
        model.fit(F, y)

        alpha[i] = model.intercept_
        B[i, :]  = model.coef_
        resid    = y - model.predict(F)
        eps_var[i] = np.var(resid, ddof=1)

        # Compute R²
        SSR = np.sum(resid**2)
        SST = np.sum((y - y.mean())**2)
        R2  = 1 - SSR / SST

        # count only non-zero factors
        p_eff = np.count_nonzero(model.coef_)

        # adjusted R² penalizes only the actually used predictors
        adj_R2[i] = 1 - (1 - R2) * (T - 1) / (T - p_eff - 1)

    # Expected returns
    mu = alpha + B.dot(f_mean)          # (n,)

    # Covariance
    Q  = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)  # (n, n)
    # ----------------------------------------------------------------------

    return mu, Q, adj_R2


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

    lambda_ = 0.0000001
    K = 2

    mu, Q = LASSO(returns, factRet, lambda_, K)
    print("Expected Returns:", mu)
    print("Covariance Matrix:", Q)
