import numpy as np

def interior_point(X, y, lam):
    """
    solve lasso using an interior point method
    requires cvxmod (Jacob Mattingley and Stephen Boyd)
    http://cvxmod.net/
    """
    import cvxmod as cvx
    n, m = X.shape
    X_cvx = cvx.matrix(np.array(X))
    y_cvx = cvx.matrix(np.array(y))
    theta = cvx.optvar('theta', m)
    p = cvx.problem(cvx.minimize(cvx.sum(cvx.atoms.power(X_cvx*theta - y_cvx, 2)) + 
                    (2*lam)*cvx.norm1(theta)))
    p.solve() 
    return np.array(cvx.value(theta))

def coordinate_descent(X, y, lam, theta=None, maxit=10000, eta=1e-8):
    """
    solve lasso using pathwise coordinate optimization
    """
    
    n, m = X.shape
    
    # initialize the coefficients
    if theta is None: theta = np.random.randn(m, 1)
    
    # compute squared columns
    v = np.diag(np.dot(X.T, X))
    
    # coordinate optimization
    i = 0
    chg = 1
    theta_old = np.empty((m, 1))
    Xtheta = np.dot(X, theta)
    while i < maxit and chg > eta:
        i += 1
        theta_old = theta.copy()
        for j in range(m):    
            Xtheta -= theta[j]*np.atleast_2d(X[:,j]).T
            alpha = np.dot(X[:,j].T, y - Xtheta)
            if abs(alpha) > lam: 
                theta[j] = np.sign(alpha) * (abs(alpha) - lam) / v[j]
                Xtheta += theta[j]*np.atleast_2d(X[:,j]).T
            else:
                theta[j] = 0
        chg = np.sum((theta - theta_old)**2)/np.sum(theta_old**2)
    
    return theta, i

def lars(X, y, mu, verbose=False):
    """
    solve lasso using lars
    """
    import path
    
    n, m = X.shape
    nz = []
    theta = np.zeros((m, 1))
    
    # find first active feature
    b = np.dot(X.T, y)
    mumax = abs(b).max()
    if mu > mumax:
        nbr = 0
        return theta, nbr
        
    # compute the homotopy starting from the first active feature
    nz.append(abs(b).argmax())
    K = 1 / (np.dot(X[:, nz].T, X[:, nz]))
    v1 = np.atleast_2d(np.sign(np.dot(X[:,nz].T, y)))
    theta_nz, nz, K, nbr = path.regularize_bwd(X, y, mumax-1e-10, mu, v1, nz, K, verbose=verbose)
    
    return theta_nz, nz, K, nbr

def one_observation(X, y, lam):
    """
    solve lasso with one observation
    """
    i0 = abs(X).argmax()
    yX = y*X[i0]
    v = np.sign(yX)
    nz = []
    if abs(yX) > lam:
        nz.append(i0)
        theta_nz = np.atleast_2d((yX - lam*v) / (X[i0]**2))
        K = np.atleast_2d(1 / (X[i0]**2))
    else:
        K = np.array([]).reshape(0,0)
        theta_nz = np.array([]).reshape(0,1)
    return theta_nz, nz, K

def fix_sol(X, y, lam, theta, thresh=1e-4):
    """
    fix the lasso solution obtained by the interior point method
    the solution from the interior point method does not identify which features are
    active unless the coefficients are thresholded
    """
    from scipy import linalg
    # identify the non-zero coefficients
    n, m = X.shape
    nz = np.where(abs(theta) > thresh)[0].tolist()
    X_nz = X[:, nz]    
    z = np.setdiff1d(np.arange(m), nz)
    X_z = X[:, z]
    
    # test if the solution is 0
    if len(nz) == 0:
        thetasol = np.zeros((m, 1))
        K = 0.
        v2 = (1/lam) * np.dot(X_z.T, y)
        truesol = True
        if abs(v2).any() > 1: truesol = False
        return thetasol, nz, K, truesol
        
    K = linalg.inv(np.dot(X_nz.T, X_nz))
    v1 = np.sign(theta[nz])
    
    # recompute the solution according to the support set and signs
    theta_nz = np.dot(K, np.dot(X_nz.T, y) - lam*v1)
    
    # look if there are other active coefficients
    v2 = (1/lam) * np.dot(X_z.T, y - np.dot(X_nz, theta_nz))
    
    if (np.sign(theta_nz) - v1).any() != 0 or abs(v2).any() > 1: truesol = False
    else: truesol = True
    
    thetasol = np.zeros((m, 1))
    thetasol[nz] = theta_nz
    return thetasol, nz, K, truesol

