import numpy as np
import cvxpy as cvx
from scipy import sparse
from time import time


def project(W, **kw):
    verbose = kw.get('verbose', False)
    M = len(W)
    V = cvx.Variable(M)
    p = cvx.Problem(cvx.Minimize(cvx.norm2(V-W)), [cvx.sum_entries(V) == 1., V > 0.])
    try:
        p.solve(max_iters=100000, verbose=verbose)
    except:
        p.solve(solver="SCS", max_iters=100000, verbose=verbose)
    w = np.array(p.variables()[0].value).flatten()
    while w.min() < 0.:
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

def maxRegularizer(W):
    return W.max(),1.*(W == W.max())
    
class maxRegularizer():
    def __init__(self, maxent=None):
        pass
    def __call__(self, W):
        return W.max(),1.*(W == W.max())
    
    
class l2Regularizer():
    def __init__(self, maxent=None):
        pass
    def __call__(self, W):
        R = np.square(np.linalg.norm(W, 2))
        return R, 2*W

class meanConstrainedRegularizer():
    def __init__(self, maxent=None):
        pass
    def __call__(self, W):
        m = len(W)
        R = np.square(np.linalg.norm(W - np.mean(W), 2))
        return R, 2*(W - np.mean(W)) * (1 + 1./float(m))


class l1JPDRegularizer():
    def __init__(self, maxent):
        self.maxent = maxent
    def __call__(self, W):
        W = sparse.csr_matrix(W)
        R = np.sum(np.abs(self.J.data))
        grad = (self.mask*self.mask.T).diagonal()
        grad = grad/np.linalg.norm(grad, 2)
        return R, grad

class l2JPDRegularizer():
    def __init__(self, maxent):
        self.maxent = maxent
    def __call__(self, W):
        W = sparse.csr_matrix(W)
        R = np.sum(np.square(self.maxent.J.data))
        grad = 2.*(self.maxent.mask*self.maxent.J*self.maxent.mask.T).diagonal()
        grad = grad/np.linalg.norm(grad, 2)
        return R, grad

class maxent():
    """A class for maximizing regularized joint entropy of MSAs

    Parameters
    ----------
    mtx : numpy.ndarray
        array representing a protein multiply sequence alignment. 
        see nmi.binMatrix. mtx.shape == (num_seqs, num_residues)
        residues
    verbose : bool, optional
        Print updates at each gradient iteration. Defaults to False
    alpha : float, optional
        The initial learning rate for opimization. Defaults to 0.1.
    rho : float, optional
        The amount of l1 regularization to use. Defaults to 0.
    wo : np.ndarray, optional
        Initial sequence weights for warm starts. len(wo) == num_seqs
    gaps : bool, optional
        Weather to consider gaps as part of the joint probability space. 
        Defaults to False.

    Attributes
    ----------
        gaps : bool
            setting for whether to consider gap characters during entropy maximization
        W : list
            list of sequence weights. appended each time gradient_step() is called. 
        objective : list
            list of objective values corresponding to W. 
        verbose : bool
            Whether to print updates during gradient optimization. 
        alpha : float
            learning rate. updated during gradient step as necessary. 
        rho : float
            rho dictates the amount of l1 regularization to apply; it is [0, 1)
        iterations : int
            number of gradient ascent iterations performed. 
    """

    def __init__(self, mtx, **kw):
        gaps = kw.get('gaps', False)
        self.mask= get_sparse_mask(mtx, gaps=gaps)
        M = self.mask.shape[0]
        self.verbose = kw.get('verbose', False)
        self.W = [kw.get('wo', np.ones(M)/float(M))]
        self.J = None
        self.alpha = kw.get('alpha', 1e-1)
        self.rho = kw.get('rho', None)  #Strength of L2 Regularization
        self.iterations = 0
        self.regularizer = None

        #Normalize the learning rate
        C = float((self.mask.T*self.mask).size)
        if self.rho is not None:
            self.alpha = self.alpha/((1.-self.rho)*C)
            self.regularizer = kw.get('regularizer', l2Regularizer)
            self.regularizer = self.regularizer(self)
        else:
            self.alpha = self.alpha/C

        self.objective = [self()]

    def gradient_step(self):
        """
        Take a gradient step from the current last recorded parameters (self.W[-1]). 
        After the step, update self.objective, self.W. If the gradient step does not
        increase the objective function value, decrement self.alpha = self.alpha/2.
        """
        n = 3 #Granularity of line search
        grad = self.gradient()
        #grad = grad/np.linalg.norm(grad, 2)
        W = project(self.W[-1] + grad)
        A = np.linspace(0., 1., n+2)
        Objective = map(self, [(1. - a)*self.W[-1] + a*W for a in A])
        a = A[np.argmax(Objective)]
        W = (1. - a)*self.W[-1] + a*W
        obj = np.max(Objective)
        self.objective.append(obj)
        self.W.append(W)
        self.iterations += 1

    def gradient_ascent(self, **kw):
        """
        Parameters
        ----------
        maxiter : int, optional
            Max number of gradient ascent iterations to undergo; default is 100
        convergence_threshold : float, optional
            Fractional difference between objective function before and 
            after a gradient step. Once this is attained, stop iteration. 
            The default is 1e-4
        """
        #C is a normalization constant for the learning rate
        #it's an attempt to make the parameters close to universal
        maxiter = kw.get('maxiter', 100)
        convergence_threshold = kw.get('convergence_threshold', 1e-4)
        start = time()
        if self.verbose:
            print "\tInitial objective value = {}".format(self.objective[-1])
        for i in range(maxiter):
            if self.verbose:
                print "Entering gradient descent cycle {}/{}".format(i+1, maxiter)
            self.gradient_step()
            diff = np.abs((self.objective[-2] - self.objective[-1])/self.objective[-2])
            if self.verbose:
                print "\tCycle {} complete, {}% diff, objective = {}".format(i+1, 100.*diff, self.objective[-1])
                print "\t{} s elapsed".format(time() - start)
            if diff <= convergence_threshold:
                print "Convergence threshold attained. Stopping iteration."
                break

    def gradient(self, W=None): 
        """
        Return the gradient of the objective function with the current weights.

        Parameters
        ----------
        W : np.ndarray, optional
            Compute the value of the objective function for a different set of weights.

        Returns
        -------
        grad : np.ndarray
            The gradient with respect to the sequence weights
        """
        if W is None:
            W = sparse.csr_matrix(np.diag(self.W[-1]))
        else:
            W = sparse.csr_matrix(np.diag(W))
        J = self.mask.T*W*self.mask
        self.J = J
        J.data = 1. + np.log(J.data)
        grad = -(self.mask*J*self.mask.T).diagonal()
        grad = grad/np.linalg.norm(grad, 2)
        if self.rho is not None:
            W = W.diagonal()
            reg = self.regularizer(W)[1]
            grad = (1. - self.rho)*grad + self.rho*reg
        return grad

    def __call__(self, W=None):
        """
        Return the value of the objective function with the current weights.

        Parameters
        ----------
        W : np.ndarray, optional
            Compute the value of the objective function for a different set of weights.

        Returns
        -------
        obj : float
            The value of the objective function
        """
        if W is None:
            W = sparse.csr_matrix(self.W[-1])
        else:
            W = sparse.csr_matrix(W)
        J = self.mask.T*(self.mask.multiply(W.T))
        self.J = J
        LogJ = J.copy()
        LogJ.data = np.log(LogJ.data)
        obj = (-J.multiply(LogJ)).sum()
        if self.rho is not None:
            obj = (1. - self.rho)*obj - self.rho*self.regularizer(W.todense())[0]
        return obj

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
