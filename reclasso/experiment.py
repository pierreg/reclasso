import numpy as np
from time import time as now
import algo
from __init__ import reclasso

def compare_algos(n=250, m=100, d=25, N=10, mu=.1, sigma=0.01, verbose=True, showfig=False, simuname='test'):
    """
    batch experiment
    compare performance of reclasso, lars, and coordinate descent with warm start
    """
    if not showfig:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    # sample observations
    thetaopt = np.zeros((m, 1))
    thetaopt[:d] = 1.
    
    # regularization parameter schedule
    lam = mu * np.arange(1, n+1)
    
    # create arrays to store the results
    br1 = np.empty((n-1, N))            # nb of transition points in the first step
    br2 = np.empty((n-1, N))            # nb of transition points in the second step
    brlars = np.empty((n-1, N))         # nb of transition points in lars
    err_hamming = np.empty((n-1, N))    # hamming distance between the current solution and hte optimal solution
    err_l2 = np.empty((n-1, N))         # l2 distance between the current solution and hte optimal solution
    t_h = np.empty((n-1, N))            # timing for the homorotpy
    t_lars = np.empty((n-1, N))         # timing for lars
    t_cd = np.empty((n-m/2, N))         # timing for coordinate descent + warm-start
    
    for j in range(N):
        
        # sample data
        X = np.random.randn(n, m)
        y = np.dot(X, thetaopt) + sigma*np.sqrt(d)*np.random.randn(n,1)
        
        # get first solution
        theta_nz, nz, K = algo.one_observation(X[0,:], y[0], lam[0])
        theta = np.zeros((m, 1))
        theta[nz] = theta_nz
        
        for i in range(1, n):
            
            if verbose: print '\nsimulation %d, observation %d'%(j, i)
            
            # solve using coordinate descent + warm-start
            if i >= (m/2):
                t0_cd = now()
                theta_cd, niter_cd = algo.coordinate_descent(X[:i+1, :], y[:i+1], lam[i], theta=theta.copy())
                t_cd[i-m/2, j] = now() - t0_cd    
                
            # solve recursive lasso
            t0_h = now()
            theta_nz, nz, K, nbr1, nbr2 = reclasso(X[:i+1, :], y[:i+1], lam[i-1], lam[i], theta_nz, nz, K, verbose=verbose)
            t_h[i-1, j] = now() - t0_h
            br1[i-1, j] = nbr1
            br2[i-1, j] = nbr2
            
            # update current solution using the homotopy solution
            theta = np.zeros((m, 1))
            theta[nz] = theta_nz
            
            # solve using lars
            t0_lars = now()
            thetalars_nz, nzlars, Klars, nbrlars = algo.lars(X[:i+1, :], y[:i+1], lam[i], verbose=verbose)
            t_lars[i-1, j] = now() - t0_lars
            brlars[i-1, j] = nbrlars
            thetalars = np.zeros((m, 1))
            thetalars[nzlars] = thetalars_nz
            
            # look at the error terms
            err_l2[i-1, j] = np.sqrt(((theta - thetaopt)**2).sum()) / np.sqrt((thetaopt**2).sum())
            err_hamming[i-1, j] = len(np.where((abs(thetaopt) > 0) != (abs(theta) > 0))[0])
            if i >= (m/2):
                # verify that the solution from coordinate descent is correct
                error = np.sqrt(((theta_cd - theta)**2).sum()) / np.sqrt((theta**2).sum())
                if error > 1e-3: print '\ncoordinate descent has not converged'
            
        # verify the result
        if verbose: 
            print '\ncompute solution using interior point method'
            thetaip = algo.interior_point(X, y, lam[-1])
            thetaip, nzip, Kip, truesol = algo.fix_sol(X, y, lam[-1], thetaip)
            print '\ntheta (ip) =', 
            print thetaip.T
            print '\ntheta (homotopy), error = ', np.sqrt(((theta - thetaip)**2).sum()) / np.sqrt((thetaip**2).sum())
            print theta.T
            print '\ntheta (lars), error = ', np.sqrt(((thetalars - thetaip)**2).sum()) / np.sqrt((thetaip**2).sum())
            print thetalars.T
            
    fig1 = plt.figure(1)
    plt.clf()
    ax = plt.subplot(111)
    brmean = (br1+br2).mean(axis=1)
    brstd = (br1+br2).std(axis=1)
    brlarsmean = (brlars).mean(axis=1)
    brlarsstd = (brlars).std(axis=1)
    iters = np.arange(1, n)
    p1 = plt.plot(iters, brmean)
    p2 = plt.plot(iters, brlarsmean)
    verts1 = zip(iters, brmean+brstd) + zip(iters[::-1], brmean[::-1] - brstd[::-1])
    verts2 = zip(iters, brlarsmean+brlarsstd) + zip(iters[::-1], brlarsmean[::-1] - brlarsstd[::-1])
    plt.legend((p1[0], p2[0]), ('Reclasso', 'Lars'), loc='upper right')
    poly1 = Polygon(verts1, facecolor=(.5,.5,1), edgecolor='w', alpha=.4, lw=0.)
    ax.add_patch(poly1)
    poly2 = Polygon(verts2, facecolor=(.5,1,.5), edgecolor='w', alpha=.4, lw=0.)
    ax.add_patch(poly2)
    plt.xlim(1, n-1)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Number of transition points', fontsize=16)
    if showfig: plt.show()
    fig1.savefig(simuname+'_transitionpoints.pdf')
    
    fig2 = plt.figure(2)
    plt.clf()
    ax = plt.subplot(111)
    hammean = err_hamming.mean(axis=1)
    hamstd = err_hamming.std(axis=1)
    p = plt.plot(iters, hammean)
    verts = zip(iters, hammean+hamstd) + zip(iters[::-1], hammean[::-1] - hamstd[::-1])
    poly = Polygon(verts, facecolor=(.5,.5,1), edgecolor='w', alpha=.4, lw=0.)
    ax.add_patch(poly)
    plt.xlim(1, n-1)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Hamming distance', fontsize=16)
    if showfig: plt.show() 
    fig2.savefig(simuname+'_hammingdistance.pdf')
    
    fig3 = plt.figure(3)
    plt.clf()
    ax = plt.subplot(111)
    l2mean = err_l2.mean(axis=1)
    l2std = err_l2.std(axis=1)
    p = plt.plot(iters, l2mean)
    verts = zip(iters, l2mean+l2std) + zip(iters[::-1], l2mean[::-1] - l2std[::-1])
    poly = Polygon(verts, facecolor=(.5,.5,1), edgecolor='w', alpha=.4, lw=0.)
    ax.add_patch(poly)
    plt.xlim(1, n-1)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Relative MSE', fontsize=16)
    if showfig: plt.show() 
    fig3.savefig(simuname+'_relativel2.pdf')
    
    fig4 = plt.figure(4)
    plt.clf()
    iters_cd = np.arange(m/2, n)
    ax = plt.subplot(111)
    t_hmean = t_h.mean(axis=1)
    t_hstd = t_h.std(axis=1)
    t_larsmean = t_lars.mean(axis=1)
    t_larsstd = t_lars.std(axis=1)
    t_cdmean = t_cd.mean(axis=1)
    t_cdstd = t_cd.std(axis=1)
    p_h = plt.plot(iters, t_hmean)
    p_lars = plt.plot(iters, t_larsmean)
    p_cd = plt.plot(iters_cd, t_cdmean)
    plt.legend((p_h[0], p_lars[0], p_cd[0]), ('Reclasso', 'Lars', 'CD'), loc='best')
    verts_h = zip(iters, t_hmean+t_hstd) + zip(iters[::-1], t_hmean[::-1] - t_hstd[::-1])
    verts_lars = zip(iters, t_larsmean+t_larsstd) + zip(iters[::-1], t_larsmean[::-1] - t_larsstd[::-1])
    verts_cd = zip(iters_cd, t_cdmean+t_cdstd) + zip(iters_cd[::-1], t_cdmean[::-1] - t_cdstd[::-1])
    poly_h = Polygon(verts_h, facecolor=(.5,.5,1), edgecolor='w', alpha=.4, lw=0.)
    ax.add_patch(poly_h)
    poly_lars = Polygon(verts_lars, facecolor=(.5,1,.5), edgecolor='w', alpha=.4, lw=0.)
    ax.add_patch(poly_lars)
    poly_cd = Polygon(verts_cd, facecolor=(1,.5,.5), edgecolor='w', alpha=.4, lw=0.)
    ax.add_patch(poly_cd)
    plt.xlim(1, n-1)
    ymax = 1.2 * max((t_larsmean+t_larsstd).max(), (t_hmean+t_hstd).max())
    plt.ylim(0, ymax)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Time', fontsize=16)
    if showfig: plt.show()
    fig4.savefig(simuname+'_timinginfo.pdf')



def adaptive_regularization(n=250, m=100, d=25, beta=0.1, lam0=.5, eta=.01, verbose=True, showfig=True):
    """
    adaptive selection of the threshold
    """
    from time import time as now
    if not showfig:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    # sample observations
    thetaopt = np.zeros((m, 1))
    perm = np.random.permutation(d)
    thetaopt[perm[:d/2]] = 1.
    thetaopt[perm[d/2:]] = -1.
    
    # sample data
    X = np.random.randn(n, m)
    y = np.dot(X, thetaopt) + beta*np.sqrt(d)*np.random.randn(n, 1)
    
    # get first solution
    lam = lam0
    theta_nz, nz, K = algo.one_observation(X[0,:], y[0], lam)
    theta = np.zeros((m, 1))
    theta[nz] = theta_nz
    
    alllam= [lam]
    br1 = []
    br2 = [] 
    err_l2 = [] 
    err_hamming = []
    
    for i in range(1, n):
        
        if verbose: print '\nobservation %d'%(i)
        
        # find new regularization parameter
        xnew = np.atleast_2d(X[i, :]).T
        ynew = y[i]
        if len(nz) > 0: dlam = float(eta*2*i*np.dot(xnew[nz].T, np.dot(K, np.sign(theta_nz)))*(np.dot(xnew[nz].T, theta_nz) - ynew))
        else: dlam = 0
        # lam_new = lam + dlam
        lam_new = lam * np.exp(lam*dlam)
        if lam_new < 0: lam_new = .0001
        alllam.append(lam_new)
        
        # solve recursive lasso
        theta_nz, nz, K, nbr1, nbr2 = reclasso(X[:i+1, :], y[:i+1], i*lam, (i+1)*lam_new, theta_nz, nz, K, verbose=verbose)
        lam = lam_new
        br1.append(nbr1)
        br2.append(nbr2)
        
        # update current solution using the homotopy solution
        theta = np.zeros((m, 1))
        theta[nz] = theta_nz
        
        # look at the error terms
        err_l2.append(np.sqrt(((theta - thetaopt)**2).sum()) / np.sqrt((thetaopt**2).sum()))
        err_hamming.append(len(np.where((abs(thetaopt) > 0) != (abs(theta) > 0))[0]))
        
    iters = np.arange(1, n)
    fig1 = plt.figure(1)
    plt.clf()
    plt.subplot(221)
    plt.plot(np.arange(n), alllam)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\lambda$')
    plt.subplot(222)
    plt.plot(iters, err_l2)
    plt.xlabel('Iteration')
    plt.ylabel('Relative MSE')
    plt.subplot(223)
    plt.plot(iters, err_hamming)
    plt.xlabel('Iteration')
    plt.ylabel('Hamming distance')
    plt.subplot(224)
    plt.stem(np.arange(m), theta)
    plt.xlabel(r'$\theta$')
    if showfig: plt.show()

