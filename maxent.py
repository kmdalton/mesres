from multiprocessing import Pool,cpu_count
import numpy as np
import cvxpy as cvx
from scipy import sparse
from time import time

def project(W, **kw):
    verbose = kw.get('verbose', False)
    bound = kw.get('bound', 0.) 
    M = len(W)
    V = cvx.Variable(M)
    p = cvx.Problem(cvx.Minimize(cvx.norm2(V-W)), [cvx.sum_entries(V) == 1., V >= bound])
    try:
        p.solve(max_iters=100000, verbose=verbose)
    except:
        p.solve(solver="SCS", max_iters=100000, verbose=verbose)
    w = np.array(p.variables()[0].value).flatten()
    if w.min() < 0.:
        w[w < 0.] = 0.
        w = project(w)
    return w


def get_sparse_mask(mtx, **kw):
    gaps = kw.get('gaps', True)
    m,l = mtx.shape
    k = mtx.max()+1
    if gaps == False:
        k = k-1
    A = sparse.csr_matrix(np.hstack((1.*(mtx==i) for i in range(k))))
    return A

class maxent():
    def __init__(self, mtx, **kw):
        gaps = kw.get('gaps', False)
        self.mask= get_sparse_mask(mtx, gaps=gaps)
        self.M = self.mask.shape[0]

    def gradient_descent(self, **kw):
        #C is a normalization constant for the learning rate
        #it's an attempt to make the parameters close to universal
        C = float((self.mask.T*self.mask).size)
        verbose = kw.get('verbose', False)
        bound = kw.get('bound', 0.)
        W = kw.get('wo', np.ones(self.M)/float(self.M))
        alpha = kw.get('alpha', 1e-2)
        accuracy_cutoff = kw.get('accuracy', 1e-3) #Fraction of change in an iteration
        maxiter = kw.get('maxiter', 100)
        rho = kw.get('rho', 0.) #Strength of L2 Regularization
        alpha = alpha/((1.-rho)*C)
        objective,params = [],[]
        start = time()
        if verbose:
            print "W is initialized to: {}".format(W)
            print "\tInitial objective value = {}".format(self(W))
        for i in range(maxiter):
            if verbose:
                print "Entering gradient descent cycle {}/{}".format(i+1, maxiter)
            W = W + alpha*((1-rho)*self.gradient(W) - rho*2.*W)
            if verbose:
                print "Projecting gradient step with cvx ..."
            W = project(W, bound=bound)
            #if verbose:
            #    print "W is: {}".format(W)
            objective.append(self(W) - np.sum(np.square(W)))
            params.append(W)
            if verbose:
                print "\tCycle {} complete, objective = {}".format(i+1, objective[-1])
                print "\t{} s elapsed".format(time() - start)
            if len(objective) > 2:
                diff = np.abs(objective[-1] - objective[-2])/objective[-2]
                if diff <= accuracy_cutoff:
                    break
        return np.array(objective), np.array(params)

    def gradient(self, W): 
        W = sparse.csr_matrix(np.diag(W))
        J = self.mask.T*W*self.mask
        J.data = np.log(J.data)
        J = J.todense()
        return -np.array([np.sum(J[(a.T*a).nonzero()] + 1) for a in self.mask])

    def __call__(self, w):
        W = sparse.csr_matrix(w)
        J = self.mask.T*(self.mask.multiply(W.T))
        LogJ = J.copy()
        LogJ.data = np.log(LogJ.data)
        return (-J.multiply(LogJ)).sum()


class hessian_maker():
    """
    COMPLETELY EXPERIMENTAL 
    NONFUNCTIONAL CODE
    """
    def __init__(self, mask):
        self.A = mask

    def __call__(self, W): 
        M,L = self.A.shape
        W = sparse.csr_matrix(np.diag(W))
        H = np.zeros((M,M))
        J = A.T*W*A
        J.data = np.log(J.data)
        J = J.todense()
        for i,a in enumerate(A):
            for j,b in enumerate(A):
                H[i,j] = J[((a.T*a).multiply(a.T*a)).nonzero()]
        return H

def shrink_psd(C, n=100):
    l,l = C.shape
    shrunk = lambda alpha: (1. - alpha)*C + alpha*np.identity(l)
    X = np.linspace(0., 1., n)
    Y = np.array([np.linalg.eig(shrunk(i))[0].min() for i in X])
    return shrunk(X[Y>0.][0])
