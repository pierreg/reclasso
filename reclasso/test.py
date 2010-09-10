from nose.tools import assert_almost_equals
import numpy as np
import algo
from __init__ import reclasso

def test_reclasso(n=5, m=5, mu0=.5, mu1=.8, thresh=1e-4, verbose=True):  
    """
    test the reclasso algorithm using random data
    """
    
    # sample a problem
    X = np.random.randn(n+1, m)
    y = np.random.randn(n+1, 1)
    
    # solve both problems using an interior point method
    theta0 = algo.interior_point(X[:-1, :], y[:-1], mu0)
    theta1 = algo.interior_point(X, y, mu1)
    
    # prepare warm-start solution for the homotopy 
    theta0, nz0, K0, truesol0 = algo.fix_sol(X[:-1, :], y[:-1], mu0, theta0, thresh=thresh)
    theta1, nz1, K1, truesol1 = algo.fix_sol(X, y, mu1, theta1, thresh=thresh)
    
    if not truesol0 or not truesol1: raise NameError, "bad threshold for interior point solution"
    
    # solve the problem using reclasso
    theta_nz, nz, K, nbr1, nbr2 = reclasso(X, y, mu0, mu1, theta0[nz0], nz0, K0, verbose=verbose, showpath=False)
    theta = np.zeros((m, 1))
    theta[nz] = theta_nz
    
    # check the result is the same as with the interior point method
    error = np.sum((theta1 - theta)**2)/np.sum(theta1**2)
    
    assert_almost_equals(error, 0)
