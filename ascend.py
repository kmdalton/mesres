import nmi,maxent
import numpy as np
from sys import argv
from os import mkdir

inFN = argv[1]
gapthresh = 0.03 #Fraction of gaps to allow in a sequence
alpha = 1e-1 #Starting learning rate
gaps = False
maxiter = 200 #I'll be shocked if > 100 steps are necessary
h,s = nmi.importFasta(inFN)
ats,mtx = nmi.prunePrimaryGaps(nmi.binMatrix(s))
m,l  = mtx.shape
#ngaps = np.sum(mtx==20, 1)/float(m)
#mtx = mtx[ngaps <= gapthresh]
#ats,mtx = nmi.prunePrimaryGaps(mtx, cutoff=0.15)
#m,l  = mtx.shape
#consensus = nmi.consensus(mtx)
#ats,mtx = ats[consensus != 20],mtx[:,consensus != 20]


#for rho in [1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]:
for rho in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#for rho in [0.]:
    outdir = str(rho)
    mkdir(outdir)
    M = maxent.maxll(mtx, verbose=True, rho=rho, alpha=alpha, gaps=gaps, regularizer=maxent.l1JPDRegularizer)

    m,l = M.mask.shape
    W = np.ones(m)/float(m)

    np.save(outdir + '/mtx', mtx)
    np.save(outdir + '/ats', ats)

    M.gradient_ascent(maxiter=maxiter, convergence_threshold = 1e-4)
    np.save(outdir + '/objective', np.array(M.objective))
    np.save(outdir + '/params', np.array(M.W))
