import numpy as np
from sklearn.linear_model import Lasso

def LASSO(returns, factRet, lambda_, K):
    """
    % Use this function for the LASSO model. Note that you will not use K
    % in this model (K is for BSS).
    %
    % You should use an optimizer to solve this problem. Be sure to comment
    % on your code to (briefly) explain your procedure.


    """

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------
     # 1) Align on dates & drop missing
    data = returns.join(factRet, how='inner').dropna()

    # 2) Factor matrix F (T×8) and compute its mean/covariance
    factor_cols = ['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev']
    F = data[factor_cols].values
    T, p = F.shape
    f_mean = F.mean(axis=0)
    Sigma_f = np.cov(F, rowvar=False, ddof=1)

    assets = returns.columns
    N = len(assets)

    # Storage
    alpha   = np.zeros(N)
    B       = np.zeros((N, p))
    eps_var = np.zeros(N)

    # 3) Fit a Lasso for each asset
    for i, asset in enumerate(assets):
        y = data[asset].values
        model = Lasso(alpha=lambda_, fit_intercept=True, max_iter=10000)
        model.fit(F, y)

        alpha[i]   = model.intercept_
        B[i, :]    = model.coef_
        resid      = y - model.predict(F)
        eps_var[i] = np.var(resid, ddof=1)

    # 4) Expected returns: 
    mu = alpha + B.dot(f_mean)          # n x 1 vector of asset exp. returns
    # 5) Covariance: 
    Q  = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)         # n x n asset covariance matrix
    
    # ----------------------------------------------------------------------

    return mu, Q
