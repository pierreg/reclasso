import numpy as np
from matplotlib.mlab import find

def reclasso(X, y, mu0, mu1, theta1, nz, K, verbose=False, showpath=False):
    """
    recursive lasso
    X           dependent variables
    y           variables
    theta1      non-zero values of the solution after n iterations
    nz          support of theta0
    K           keep matrix for rank 1 updates
    """
    import path
    
    # Step 1: vary the regularization parameter from mu0 to mu1
    if mu0 < mu1:
        theta1, nz, K, nbr1 = path.regularize_fwd(X[:-1, :], y[:-1], mu0, mu1, np.sign(theta1), nz, K, verbose=verbose, showpath=showpath)
    elif mu0 > mu1:
        theta1, nz, K, nbr1 = path.regularize_bwd(X[:-1, :], y[:-1], mu0, mu1, np.sign(theta1), nz, K, verbose=verbose)
    else:
        nbr1 = 0
    
    # Step 2: vary the parameter t from 0 to 1
    theta1, nz, K, nbr2 = path.add_observation(X, y, mu1, theta1, nz, K, verbose=verbose, showpath=showpath)
    
    return theta1, nz, K, nbr1, nbr2


