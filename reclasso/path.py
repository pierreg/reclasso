import numpy as np

def invupdateapp(A, x, y, r):
    """update the inverse of a matrix appending one column and one row"""
    yA = np.dot(y, A)
    q = 1 / (r - np.dot(yA, x))
    Ax = q * np.dot(A, x)
    return np.vstack([np.hstack([A + np.dot(Ax, yA), -Ax]), np.hstack([-yA * q, q])])

def invupdatered(A, c):
    """update the inverse of a matrix reducing one column and one row"""
    n, m = A.shape
    indn = np.arange(n)
    q = A[c, c]
    c1 = np.hstack((indn[:c], indn[c+1:]))
    Ax = np.atleast_2d(A[c1, c])
    yA = np.atleast_2d(A[c, c1])
    return A[c1][:,c1] - np.dot(Ax.T, yA)/q

def regularize_fwd(X, y, mu0, mu1, v1, nz, K, verbose=False, showpath=False, fignum=1):
    """
    compute the solution path when the regularization varies between mu0 and mu1
    mu1 > mu0
    """
    
    if verbose: print '\ncompute path between mu=%.4f and mu=%.4f'%(mu0, mu1)
    
    n, m = X.shape
    X_nz = np.atleast_2d(X[:, nz])
    b = np.dot(X.T, y)
    G = np.dot(X.T, X)
    
    nbr = 0
    mu = mu0
    trans_type = -1
    trans_sign = 0
    trans_ind = -1
    if verbose: print 'initial active features =', nz
    if showpath:
        import matplotlib.pyplot as plt
        pth = np.linspace(mu0, mu1, 100)
        thetapth = np.zeros((m, 100))
        fig = plt.figure(fignum)
        plt.clf()
        allbr = []
        
    while mu < mu1:
        
        # find the breakpoints where coefficients become zero
        b_nz = b[nz]
        Kv1 = np.dot(K, v1)
        Kb_nz = np.dot(K, b_nz)
        mu_0 = Kb_nz / Kv1
        
        # find the breakpoints where new coefficients become active
        z = np.setdiff1d(np.arange(m), nz)
        X_z = np.atleast_2d(X[:, z])
        b_z = b[z]
        M = G[np.ix_(z, nz)]
        MKb_nz = np.dot(M, Kb_nz)
        MKv1 = np.dot(M, Kv1)
        mu_1 = (b_z - MKb_nz) / (1 - MKv1)
        mu_m1 = (b_z - MKb_nz) / (-1 - MKv1)
        
        if trans_type > 0: mu_0[-1] = mu1
        mu_0[mu_0 <= mu] = mu1
        if len(mu_0) > 0:            
            mu_0_argmin = mu_0.argmin()
            mu_0_min = mu_0[mu_0_argmin][0]
        else:
            mu_0_min = mu1
        if trans_type == 0:
            if trans_sign == 1: mu_1[np.where(z == trans_ind)[0]] = mu1 + 1
            else: mu_m1[np.where(z == trans_ind)[0]] = mu1 + 1
        mu_1[mu_1 <= mu] = mu1
        if len(mu_1) > 0:            
            mu_1_argmin = mu_1.argmin()
            mu_1_min = mu_1[mu_1_argmin][0]
        else:
            mu_1_min = mu1
        mu_m1[mu_m1 <= mu] = mu1
        if len(mu_m1) > 0:            
            mu_m1_argmin = mu_m1.argmin()
            mu_m1_min = mu_m1[mu_m1_argmin][0]
        else:
            mu_m1_min = mu1
            
        # compute the breakpoint
        mu_br_all = np.array([mu_0_min, mu_1_min, mu_m1_min])
        trans_type = mu_br_all.argmin()
        mu_br = mu_br_all[trans_type]
        
        if mu_br < mu1:
            
            if showpath:
                if len(nz) > 0:
                    inds = np.intersect1d(np.where(pth >= mu)[0], np.where(pth < mu_br)[0])
                    thetapth[np.ix_(nz, inds)] = np.tile(Kb_nz, (1, len(inds))) - np.tile(Kv1, (1, len(inds))) * \
                                                 np.tile(pth[inds], (len(nz), 1))
                allbr.append(mu_br)
                
            nbr += 1
            mu = mu_br
            
            if trans_type == 0:         # an element of theta(t) goes to zero
                trans_ind = nz[mu_0_argmin]
                trans_sign = v1[mu_0_argmin]
                if verbose: print 'transition point :: mu = %.4f :: feature %d is inactive'%(mu, trans_ind)
                nzind = range(len(nz))
                nzind.remove(nz.index(trans_ind))
                v1 = v1[nzind]
                nz.remove(trans_ind)
                X_nz = X[:, nz]
                K = invupdatered(K, mu_0_argmin)
            else:                       # new active element
                if trans_type == 1:      # it is positive
                    trans_ind = z[mu_1_argmin]
                    if verbose: print 'transition point :: mu = %.4f :: feature %d is positive'%(mu, trans_ind)
                    nz.append(trans_ind)
                    v1 = np.vstack([v1, 1])
                else:                   # it is negative
                    trans_ind = z[mu_m1_argmin]
                    if verbose: print 'transition point :: mu = %.4f :: feature %d is negative'%(mu, trans_ind)
                    nz.append(trans_ind)
                    v1 = np.vstack([v1, -1])
                X_new = np.atleast_2d(X[:, trans_ind]).T
                K = invupdateapp(K, np.dot(X_nz.T, X_new), np.dot(X_new.T, X_nz), 
                                 np.dot(X_new.T, X_new))
                X_nz = X[:, nz]
                
        else:                           # compute solution at mu1
        
            if verbose: print 'compute solution at mu =', mu1
            if showpath and len(nz) > 0:
                inds = np.intersect1d(np.where(pth >= mu)[0], np.where(pth <= mu1)[0])
                thetapth[np.ix_(nz, inds)] = np.tile(Kb_nz, (1, len(inds))) - np.tile(Kv1, (1, len(inds))) * \
                                             np.tile(pth[inds], (len(nz), 1))
            
            theta_nz = Kb_nz - mu1*Kv1
            mu = mu1
    
    if showpath:
        fig = plt.figure(fignum)
        leg = []
        for i in range(m):
            plt.plot(pth, thetapth[i, :])
            leg.append(r'$\theta_%d(\mu)$'%(i+1))
        plt.plot(pth, np.zeros(len(pth),), 'k')
        plt.xlabel(r'$\mu$', fontsize=16)
        plt.title(r'Step 1: homotopy in $\mu$', fontsize=16)
        plt.legend(leg, loc='best')
        plt.plot(allbr, np.zeros(nbr), 'ko')
        plt.xlim(mu0, mu1)
        plt.show()
    
    return theta_nz, nz, K, nbr

def regularize_bwd(X, y, mu0, mu1, v1, nz, K, verbose=False):
    """
    compute the solution path when the regularization varies between mu0 and mu1
    mu1 < mu0
    """
    
    if verbose: print '\ncompute bath between mu=%.4f and mu=%.4f'%(mu0, mu1)
    
    n, m = X.shape
    X_nz = np.atleast_2d(X[:, nz])
    b = np.dot(X.T, y)
    G = np.dot(X.T, X)
    
    nbr = 0
    mu = mu0
    trans_type = -1
    trans_sign = 0
    trans_ind = -1
    if verbose: print 'initial active features =', nz
    
    while mu > mu1:
        
        # find the breakpoints where coefficients become zero
        b_nz = b[nz]
        Kv1 = np.dot(K, v1)
        Kb_nz = np.dot(K, b_nz)
        mu_0 = Kb_nz / Kv1
        
        # find the breakpoints where new coefficients become active
        z = np.setdiff1d(np.arange(m), nz)
        X_z = np.atleast_2d(X[:, z])
        b_z = b[z]
        M = G[np.ix_(z, nz)]
        MKb_nz = np.dot(M, Kb_nz)
        MKv1 = np.dot(M, Kv1)
        mu_1 = (b_z - MKb_nz) / (1 - MKv1)
        mu_m1 = (b_z - MKb_nz) / (-1 - MKv1)
        
        if trans_type > 0: mu_0[-1] = mu1
        mu_0[mu_0 >= mu] = mu1
        if len(mu_0) > 0:            
            mu_0_argmax = mu_0.argmax()
            mu_0_max = mu_0[mu_0_argmax][0]
        else:
            mu_0_max = mu1
        if trans_type == 0:
            if trans_sign == 1: mu_1[np.where(z == trans_ind)[0]] = mu1 - 1
            else: mu_m1[np.where(z == trans_ind)[0]] = mu1 - 1
        mu_1[mu_1 >= mu] = mu1
        if len(mu_1) > 0:            
            mu_1_argmax = mu_1.argmax()
            mu_1_max = mu_1[mu_1_argmax][0]
        else:
            mu_1_max = mu1
        mu_m1[mu_m1 >= mu] = mu1
        if len(mu_m1) > 0:            
            mu_m1_argmax = mu_m1.argmax()
            mu_m1_max = mu_m1[mu_m1_argmax][0]
        else:
            mu_m1_max = mu1
        
        # compute the breakpoint
        mu_br_all = np.array([mu_0_max, mu_1_max, mu_m1_max])
        trans_type = mu_br_all.argmax()
        mu_br = mu_br_all[trans_type]
        
        if mu_br > mu1:
            
            nbr += 1
            mu = mu_br
            
            if trans_type == 0:         # an element of theta(t) goes to zero
                trans_ind = nz[mu_0_argmax]
                trans_sign = v1[mu_0_argmax]
                if verbose: print 'transition point :: mu = %.4f :: feature %d is inactive'%(mu, trans_ind)
                nzind = range(len(nz))
                nzind.remove(nz.index(trans_ind))
                v1 = v1[nzind]
                nz.remove(trans_ind)
                X_nz = X[:, nz]
                K = invupdatered(K, mu_0_argmax)
            else:                       # new active element
                if trans_type == 1:      # it is positive
                    trans_ind = z[mu_1_argmax]
                    if verbose: print 'transition point :: mu = %.4f :: feature %d is positive'%(mu, trans_ind)
                    nz.append(trans_ind)
                    v1 = np.vstack([v1, 1])
                else:                   # it is negative
                    trans_ind = z[mu_m1_argmax]
                    if verbose: print 'transition point :: mu = %.4f :: feature %d is negative'%(mu, trans_ind)
                    nz.append(trans_ind)
                    v1 = np.vstack([v1, -1])
                X_new = np.atleast_2d(X[:, trans_ind]).T
                K = invupdateapp(K, np.dot(X_nz.T, X_new), np.dot(X_new.T, X_nz), 
                                 np.dot(X_new.T, X_new))
                X_nz = X[:, nz]
            
        else:                           # compute solution at mu1
        
            if verbose: print 'compute solution at mu =', mu1
            theta_nz = Kb_nz - mu1*Kv1
            mu = mu1
        
    return theta_nz, nz, K, nbr

def add_observation(X, y, mu, theta_nz, nz, K, verbose=False, showpath=False, fignum=2):
    """
    compute the solution path when t varies from 0 to 1
    """
    
    if verbose: print '\ncompute path between t=0 and t=1'
    
    n, m = X.shape
    X_nz = np.atleast_2d(X[:, nz])
    v1 = np.sign(theta_nz)
    psi = np.atleast_2d(X[-1,:]).T
    b = np.dot(X.T, y)
    
    # update K to take into account added row
    K -= (1 / (1 + np.dot(psi[nz].T, np.dot(K, psi[nz])))) * np.dot(K, np.dot(psi[nz], np.dot(psi[nz].T, K)))
    
    nbr = 0
    t = 0
    trans_type = -1
    trans_sign = 0
    trans_ind = -1
    if verbose: print 'initial active features =', nz
    if showpath:
        import matplotlib.pyplot as plt
        pth = np.linspace(0, 1, 100)
        thetapth = np.zeros((m, 100))
        fig = plt.figure(fignum)
        plt.clf()
        allbr = []
        
    while t < 1:
        
        # update various parameters
        theta_nz = np.dot(K, b[nz] - mu*v1)
        eb = np.dot(psi[nz].T, theta_nz) - y[-1]
        err = np.dot(X_nz, theta_nz) - y
        u = np.dot(K, psi[nz])
        alpha = np.dot(psi[nz].T, u)
        if len(nz) == 0:                # because of numpy bug
            alpha = 0.
            eb = -y[-1]
            
        # find the breakpoints where coefficients become zero
        tmp = 1 + (eb * u / theta_nz - alpha)**(-1)
        tmp[tmp < 0] = 1
        t_0 = tmp**.5
        
        # find the breakpoints where new coefficients become active
        z = np.setdiff1d(np.arange(m), nz)
        X_z = np.atleast_2d(X[:, z])
        v = np.dot(np.dot(X_z.T, X_nz), u)
        Xe = np.dot(X_z.T, err)
        
        tmp = 1 + (eb*(psi[z] - v)/(-mu - Xe) - alpha)**(-1)
        tmp[tmp < 0] = 1
        t_1 = tmp**.5
        tmp = 1 + (eb*(psi[z] - v)/(mu - Xe) - alpha)**(-1)
        tmp[tmp < 0] = 1
        t_m1 = tmp**.5
        
        if trans_type > 0: t_0[-1] = 1
        t_0[t_0 <= t] = 1
        if len(t_0) > 0:            
            t_0_argmin = t_0.argmin()
            t_0_min = t_0[t_0_argmin][0]
        else:
            t_0_min = 1
        if trans_type == 0:
            if trans_sign == 1: t_1[np.where(z == trans_ind)[0]] = 1
            else: t_m1[np.where(z == trans_ind)[0]] = 1
        t_1[t_1 <= t] = 1
        if len(t_1) > 0: 
            t_1_argmin = t_1.argmin()
            t_1_min = t_1[t_1_argmin][0]
        else:
            t_1_min = 1
        t_m1[t_m1 <= t] = 1
        if len(t_m1) > 0: 
            t_m1_argmin = t_m1.argmin()
            t_m1_min = t_m1[t_m1_argmin][0]
        else:
            t_m1_min = 1
            
        # compute the breakpoint
        t_br_all = np.array([t_0_min, t_1_min, t_m1_min])
        trans_type = t_br_all.argmin()
        t_br = t_br_all[trans_type]
        
        if t_br < 1:
            
            if showpath:
                if len(nz) > 0:
                    inds = np.intersect1d(np.where(pth >= t)[0], np.where(pth < t_br)[0])                    
                    thetapth[np.ix_(nz, inds)] = np.tile(theta_nz, (1, len(inds))) - np.tile(u, (1, len(inds))) * \
                                                 np.tile(eb * (pth[inds]**2 - 1) / (1 + alpha*(pth[inds]**2 - 1)), (len(nz), 1))
                allbr.append(t_br)
                
            nbr += 1
            t = t_br
            
            if trans_type == 0:         # an element of theta(t) goes to zero
                trans_ind = nz[t_0_argmin]
                trans_sign = v1[t_0_argmin]
                if verbose: print 'transition point :: t = %.4f :: feature %d is inactive'%(t, trans_ind)
                nzind = range(len(nz))
                nzind.remove(nz.index(trans_ind))
                v1 = v1[nzind]
                nz.remove(trans_ind)
                X_nz = X[:, nz]
                K = invupdatered(K, t_0_argmin)
            else:                       # new active element
                if trans_type == 1:      # it is positive
                    trans_ind = z[t_1_argmin]
                    if verbose: print 'transition point :: t = %.4f :: feature %d is positive'%(t, trans_ind)
                    nz.append(trans_ind)
                    v1 = np.vstack([v1, 1])
                else:                   # it is negative
                    trans_ind = z[t_m1_argmin]
                    if verbose: print 'transition point :: t = %.4f :: feature %d is negative'%(t, trans_ind)
                    nz.append(trans_ind)
                    v1 = np.vstack([v1, -1])
                X_new = np.atleast_2d(X[:, trans_ind]).T
                K = invupdateapp(K, np.dot(X_nz.T, X_new), np.dot(X_new.T, X_nz), 
                                 np.dot(X_new.T, X_new))
                if len(nz) == 1: K = 1 / np.dot(X_new.T, X_new)     # because of numpy bug
                X_nz = X[:, nz]
                
        else:                           # compute solution at mu1
        
            if verbose: print 'compute solution at t = 1'
            if showpath and len(nz) > 0:
                inds = np.intersect1d(np.where(pth >= t)[0], np.where(pth <= 1)[0])
                thetapth[np.ix_(nz, inds)] = np.tile(theta_nz, (1, len(inds))) - np.tile(u, (1, len(inds))) * \
                                             np.tile(eb * (pth[inds]**2 - 1) / (1 + alpha*(pth[inds]**2 - 1)), (len(nz), 1))
            t = 1
            
    if showpath:
        fig = plt.figure(fignum)
        leg = []
        for i in range(m):
            plt.plot(pth, thetapth[i, :])
            leg.append(r'$\theta_%d(t)$'%(i+1))
        plt.plot(pth, np.zeros(len(pth),), 'k')
        plt.xlabel(r'$t$', fontsize=16)
        plt.title(r'Step 2: homotopy in $t$', fontsize=16)
        plt.legend(leg, loc='best')
        plt.plot(allbr, np.zeros(nbr), 'ko')
        plt.show()
        
    return theta_nz, nz, K, nbr

def remove_observation(X, y, mu, k, theta_nz, nz, K, verbose=False):
    """
    compute the solution path when t varies from 1 to 0
    """
    
    if verbose: print '\ncompute path between t=1 and t=0'    
    
    n, m = X.shape
    psi = np.atleast_2d(X[k,:]).T
    yb = y[k]
    
    X_nz = np.atleast_2d(X[:, nz])
    v1 = np.sign(theta_nz)
    b = np.dot(X.T, y)
    
    nbr = 0
    t = 1
    trans_type = -1
    trans_sign = 0
    trans_ind = -1
    if verbose: print 'initial active features =', nz
    
    while t > 0:
        
        # update various parameters
        theta_nz = np.dot(K, b[nz] - mu*v1)
        eb = np.dot(psi[nz].T, theta_nz) - yb
        err = np.dot(X_nz, theta_nz) - y
        u = np.dot(K, psi[nz])
        alpha = np.dot(psi[nz].T, u)
        
        # find the breakpoints where coefficients become zero
        tmp = 1 + (eb * u / theta_nz - alpha)**(-1)
        tmp[tmp < 0] = 0
        t_0 = tmp**.5
        
        # find the breakpoints where new coefficients become active
        z = np.setdiff1d(np.arange(m), nz)
        X_z = np.atleast_2d(X[:, z])
        v = np.dot(np.dot(X_z.T, X_nz), u)
        Xe = np.dot(X_z.T, err)
        
        tmp = 1 + (eb*(psi[z] - v)/(-mu - Xe) - alpha)**(-1)
        tmp[tmp < 0] = 0
        t_1 = tmp**.5
        tmp = 1 + (eb*(psi[z] - v)/(mu - Xe) - alpha)**(-1)
        tmp[tmp < 0] = 0
        t_m1 = tmp**.5
        
        if trans_type > 0: t_0[-1] = 0
        t_0[t_0 >= t] = 0
        if len(t_0) > 0:            
            t_0_argmax = t_0.argmax()
            t_0_max = t_0[t_0_argmax][0]
        else:
            t_0_max = 0
        if trans_type == 0:
            if trans_sign == 1: t_1[np.where(z == trans_ind)[0]] = 0
            else: t_m1[np.were(z == trans_ind)[0]] = 0
        t_1[t_1 >= t] = 0
        if len(t_1) > 0: 
            t_1_argmax = t_1.argmax()
            t_1_max = t_1[t_1_argmax][0]
        else:
            t_1_min = 0
        t_m1[t_m1 >= t] = 0
        if len(t_m1) > 0: 
            t_m1_argmax = t_m1.argmax()
            t_m1_max = t_m1[t_m1_argmax][0]
        else:
            t_m1_max = 0
            
        # compute the breakpoint
        t_br_all = np.array([t_0_max, t_1_max, t_m1_max])
        trans_type = t_br_all.argmax()
        t_br = t_br_all[trans_type]
        
        if t_br > 0:
            
            nbr += 1
            t = t_br
            
            if trans_type == 0:         # an element of theta(t) goes to zero
                trans_ind = nz[t_0_argmax]
                trans_sign = v1[t_0_argmax]
                if verbose: print 'transition point :: t = %.4f :: feature %d is inactive'%(t, trans_ind)
                nzind = range(len(nz))
                nzind.remove(nz.index(trans_ind))
                v1 = v1[nzind]
                nz.remove(trans_ind)
                X_nz = X[:, nz]
                K = invupdatered(K, t_0_argmax)
            else:                       # new active element
                if trans_type == 1:      # it is positive
                    trans_ind = z[t_1_argmax]
                    if verbose: print 'transition point :: t = %.4f :: feature %d is positive'%(t, trans_ind)
                    nz.append(trans_ind)
                    v1 = np.vstack([v1, 1])
                else:                   # it is negative
                    trans_ind = z[t_m1_argmax]
                    if verbose: print 'transition point :: t = %.4f :: feature %d is negative'%(t, trans_ind)
                    nz.append(trans_ind)
                    v1 = np.vstack([v1, -1])
                X_new = np.atleast_2d(X[:, trans_ind]).T
                K = invupdateapp(K, np.dot(X_nz.T, X_new), np.dot(X_new.T, X_nz), 
                                 np.dot(X_new.T, X_new))
                X_nz = X[:, nz]
                
        else:                           # compute solution at mu1
        
            if verbose: print 'compute solution at t = 0'
            t = 0
            theta_nz += eb * u / (1 - alpha)
            
    return theta_nz, nz, K, nbr


