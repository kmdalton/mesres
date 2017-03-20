from multiprocessing import Pool,cpu_count
import numpy as np
import cvxpy as cvx
from scipy import sparse
from time import time

pool = Pool(cpu_count())

def project(W, **kw):
    verbose = kw.get('verbose', False)
    bound = kw.get('bound', 0.) 
    M = len(W)
    V = cvx.Variable(M)
    p = cvx.Problem(cvx.Minimize(cvx.norm2(V-W)), [cvx.sum_entries(V) == 1., V > bound])
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
        rho = kw.get('rho', None)  #Strength of L2 Regularization
        if rho is not None:
            alpha = alpha/((1.-rho)*C)
        objective,params = [],[]
        start = time()
        if verbose:
            print "W is initialized to: {}".format(W)
            print "\tInitial objective value = {}".format(self(W))
        for i in range(maxiter):
            if verbose:
                print "Entering gradient descent cycle {}/{}".format(i+1, maxiter)
            W = W + alpha*self.gradient(W, rho=rho) 
            if verbose:
                print "Projecting gradient step with cvx ..."
            W = project(W, bound=bound)
            #if verbose:
            #    print "W is: {}".format(W)
            objective.append(self(W, rho) - np.sum(np.square(W)))
            params.append(W)
            if verbose:
                print "\tCycle {} complete, objective = {}".format(i+1, objective[-1])
                print "\t{} s elapsed".format(time() - start)
            if len(objective) > 2:
                diff = np.abs(objective[-1] - objective[-2])/objective[-2]
                if diff <= accuracy_cutoff:
                    break
        return np.array(objective), np.array(params)

    def gradient(self, W, rho=None): 
        W = sparse.csr_matrix(np.diag(W))
        J = self.mask.T*W*self.mask
        J.data = 1. + np.log(J.data)
        #J = J.todense()
        #grad = -np.array([np.sum(J[(a.T*a).nonzero()] + 1) for a in self.mask])
        grad = -(self.mask*J*self.mask.T).diagonal()
        if rho is not None:
            #reg = -np.array([(2.*w*a.T*a).sum() for w,a in zip(W.diagonal(),self.mask)]) #L2
            reg = -np.array([(a.T*a).sum() for a in self.mask]) #L1
            #reg = -(self.mask*self.mask.T).diagonal() #L1
            #reg = reg*(1.*(np.array(W.diagonal()) > 0.))
            grad = (1. - rho)*grad + rho*reg
        return grad

    def pgradient(self, W):
        gmaker = gradient_maker(self.mask, W)
        grad = pool.map(gmaker, range(self.M))
        return np.array(grad)

    def __call__(self, w, rho=None):
        W = sparse.csr_matrix(w)
        J = self.mask.T*(self.mask.multiply(W.T))
        LogJ = J.copy()
        LogJ.data = np.log(LogJ.data)
        obj = (-J.multiply(LogJ)).sum()
        if rho is not None:
            obj += rho*J.sum()
        return obj

class gradient_maker():
    def __init__(self, A, W): 
        self.mask = A
        self.W = sparse.csr_matrix(np.diag(W))
        J = self.mask.T*W*self.mask
        J.data = np.log(J.data)
        self.J = J.todense()
    def __call__(self, i): 
        return -np.sum(self.J[(self.mask[i].T*self.mask[i]).nonzero()] + 1)

def shrink_psd(C, n=100):
    l,l = C.shape
    shrunk = lambda alpha: (1. - alpha)*C + alpha*np.identity(l)
    CW = None
    for alpha in np.linspace(0., 1., n):
        CW = shrunk(alpha)
        if np.linalg.eig(shrunk(alpha))[0].min() > 0.:
            break
    return CW

def joint_binmat(A, **kw):
    verbose = kw.get('verbose', False)
    m,l = A.shape
    A = A.T
    return sparse.hstack([A.multiply(a.t) for a in A.T])

def reference_hessian(A, W):
    m,l = A.shape
    H = np.zeros((m,m))
    P = A.T*np.diag(W)*A
    I = P.copy()
    I[P>0.] = 1./I[P>0.]
    for i in range(m):
        for j in range(m):
            H[i,j] = (A[i].T*A[i]).multiply(A[j].T*A[j]).multiply(I).sum()
    return H
