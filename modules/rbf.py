import numpy as np
import scipy.spatial.distance as sd
import numpy.polynomial.polynomial as pp

def rbf_fd_weights(X,ctr,s,d):
    #   X : each row contains one node in R^2
    # ctr : center (evaluation) node
    # s,d : PHS order and polynomial degree
        
    rbf  = lambda r,s: r**s
    Drbf = lambda r,s,xi: s*xi*r**(s-2)
    Lrbf = lambda r,s: s**2*r**(s-2)
        
    n = X.shape[0] 
    for i in range(2): X[:,i] -= ctr[i]
    DM = sd.squareform(sd.pdist(X))
    D0 = np.sqrt(X[:,0]**2 + X[:,1]**2)
    A = rbf(DM,s)
    b = np.vstack((Lrbf(D0,s),Drbf(D0,s,-X[:,0]),
                    Drbf(D0,s,-X[:,1]))).T
                    
    if d > -1: #adding polynomials
        m = int((d+2)*(d+1)/2)              
        O, k = np.zeros((m,m)), 0
        P, LP = np.zeros((n,m)), np.zeros((m,3))
        PX = pp.polyvander(X[:,0],d)
        PY = pp.polyvander(X[:,1],d)
        for j in range(d+1):
            P[:,k:k+j+1], k = PX[:,j::-1]*PY[:,:j+1], k+j+1
        if d > 0: LP[1,1], LP[2,2] = 1, 1
        if d > 1: LP[3,0], LP[5,0] = 2, 2 
        A = np.block([[A,P],[P.T,O]])	
        b = np.block([[b],[LP]])
    # each column contains the weights for 
    # the Laplacian, d/dx1, d/dx2, respectivly.
    weights = np.linalg.solve(A,b)
    return weights[:n,:]


def rbf_interpolator_inv_matrix(X,ctr,s,d, velocity, density):
    #   X : each row contains one node in R^2
    # ctr : center (evaluation) node
    # s,d : PHS order and polynomial degree
        
    rbf  = lambda r,s: r**s
        
    n = X.shape[0] 
    for i in range(2): X[:,i] -= ctr[i]
    DM = sd.squareform(sd.pdist(X))
    D0 = np.sqrt(X[:,0]**2 + X[:,1]**2)
    A = rbf(DM,s)
    b = np.vstack((velocity[:,0], velocity[:,1], density)).T
                    
    if d > -1: #adding polynomials
        m = int((d+2)*(d+1)/2)              
        O, k = np.zeros((m,m)), 0
        P, LP = np.zeros((n,m)), np.zeros((m,3))
        PX = pp.polyvander(X[:,0],d)
        PY = pp.polyvander(X[:,1],d)
        for j in range(d+1):
            P[:,k:k+j+1], k = PX[:,j::-1]*PY[:,:j+1], k+j+1
        if d > 0: LP[1,1], LP[2,2] = 1, 1
        if d > 1: LP[3,0], LP[5,0] = 2, 2 
        A = np.block([[A,P],[P.T,O]])	
        b = np.block([[b],[np.zeros((m,3))]])
    # each column contains the weights for 
    # the Laplacian, d/dx1, d/dx2, respectivly.
    
    return np.linalg.inv(A)