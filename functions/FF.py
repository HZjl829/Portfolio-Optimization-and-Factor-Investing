import numpy as np
import pandas as pd

def FF(returns, factRet, lambda_, K):
    """
    Calibrate the Fama-French 3-factor model.

    Returns:
      mu      : n-vector of expected returns
      Q       : n×n asset covariance matrix
      adj_R2  : n-vector of adjusted R² for each asset regression
    """
    # ----------------------------------------------------------------------
    # align dates and drop any rows with missing data
    data = returns.join(factRet[['Mkt_RF','SMB','HML']], how='inner').dropna()
    
    # build design matrix X = [1, Mkt_RF, SMB, HML]
    F = data[['Mkt_RF','SMB','HML']].values    # (T, 3)
    T = F.shape[0]
    X = np.hstack([np.ones((T, 1)), F])        # (T, 4)
    
    assets = returns.columns
    N = len(assets)
    
    # storage
    B       = np.zeros((N, 3))
    alpha   = np.zeros(N)
    eps_var = np.zeros(N)
    adj_R2  = np.zeros(N)
    
    # run OLS for each asset
    for i, asset in enumerate(assets):
        y, *_ = data[asset].values, 
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        
        alpha[i] = coeffs[0]
        B[i, :]  = coeffs[1:]
        
        resid = y - X.dot(coeffs)
        eps_var[i] = resid.var(ddof=1)
        
        # compute R²
        SSR = np.sum(resid**2)
        SST = np.sum((y - y.mean())**2)
        R2  = 1 - SSR/SST
        
        # count only nonzero betas (exclude intercept)
        p_eff = np.count_nonzero(coeffs[1:])
        
        # adjusted R² with p_eff predictors
        adj_R2[i] = 1 - (1 - R2) * (T - 1) / (T - p_eff - 1)
    
    # expected returns
    f_mean = F.mean(axis=0)             # (3,)
    mu     = alpha + B.dot(f_mean)      # (N,)
    
    # factor-model covariance
    Sigma_f = np.cov(F, rowvar=False, ddof=1)  
    Q       = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)
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

    lambda_ = 1
    K = 2

    mu, Q = FF(returns, factRet, lambda_, K)
    print("Expected Returns:", mu)
    print("Covariance Matrix:", Q)
