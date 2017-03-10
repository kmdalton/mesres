import numpy as np
import re

#Includes the gap character!
AminoAcids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']

#Maps amino acid names onto ints for the alignment matrix
aaMapping = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19,
    '-': 20,
    0 :'A',
    1 :'C',
    2 :'D',
    3 :'E',
    4 :'F',
    5 :'G',
    6 :'H',
    7 :'I',
    8 :'K',
    9 :'L',
    10:'M',
    11:'N',
    12:'P',
    13:'Q',
    14:'R',
    15:'S',
    16:'T',
    17:'V',
    18:'W',
    19:'Y',
    20:'-'
}

#ordMapping is a faster hash for constructing the alignment matrix
#hashing is by unicode code point of character.
ordMapping = 20*np.ones(500, dtype='int')
for i in AminoAcids:
    ordMapping[ord(i)] = aaMapping[i]

#Unknown amino acids default to gap character
ordMapping[ord('X')] = aaMapping['-']
ordMapping[ord(' ')] = aaMapping['-']

#Background level frequences for amino acids 
#Ranganathan's Notes on SCA -- ordered according to the hash above
#Would be interesting to see how this compares in prokaryotes versus eukaryotes
BGQ = np.array([0.073, 0.025, 0.050, 0.061, 0.042, 0.072, 0.023, 0.053, 0.064, 0.089, 0.023, 0.043, 0.052, 0.040, 0.052, 0.073, 0.056, 0.063, 0.013, 0.033])

#Takes a iterable of aligned sequences and returns the corresponding numpy array
def binMatrix(seqs):
    # keep only sequences that have the mode length
    L = np.argmax(np.bincount(np.array([len(i) for i in seqs])))
    M = len(seqs)
    mtx = 20*np.ones([M,L], dtype='int')
    for i in range(M):
        try:
            mtx[i,:len(seqs[i])] = ordMapping[[ord(j) for j in seqs[i].strip().upper()]]
        except ValueError:
            # This is called when the size of the matrix is too small
            print 'Failure parsing sequence with header: %s' %headers[i]
            print 'Sequence length: %s' %len(seqs[i])
    return mtx

# builds binary Rama matrix (20 aa, no gap)
def binMatrix3D(mtx):
    """Returns mtx (m by n) as a binary matrix n by m by 20
    with each column of the third dimension equal to zero except 
    at index mtx(i,j), where it equals unity.

    Helper function of Rama Ranganathan MATLAB sca5 function. 
    """
    mtx3d = np.zeros(mtx.shape+(20,))

    for i in range(mtx3d.shape[0]):
        for j in range(mtx3d.shape[1]):
            if mtx[i,j] != 20:
                mtx3d[i,j,mtx[i,j]] = 1.0
    return mtx3d

# python transcription of weight_aln.m 
def weightMatrix(mtx3d,bgq):
    """
    Calculation of weight tensor, which replaces unity values in
    mtx3d with the positions relative entropy. 

    Helper function of Rama Ranganathan MATLAB sca5 function. 
    """
    nseq,npos,naa = mtx3d.shape
    
    mtx3d_mat = np.reshape(mtx3d.transpose(2,1,0),(naa*npos,nseq),order='F')
    f1_v =np.sum(mtx3d_mat.T,axis=0)/nseq
    w_v = np.squeeze(np.ones((1,naa*npos)))
    q1_v = np.squeeze(np.tile(bgq,(1,npos)))

    for x in range(naa*npos):
        q = q1_v[x]
        f = f1_v[x]
        # here I hard coded their metric DerivEntropy
        if f > 0 and f < 1:
            w_v[x] = np.abs(np.log(f*(1-q)/(q*(1-f))))
        else: 
            w_v[x] = 0.

    W = np.zeros((npos,naa))
    for i in range(npos):
        for j in range(naa):
            W[i,j] = w_v[naa*i+j]
            
    Wx = np.tile(np.reshape(W,(1, npos, naa),order='F'),(nseq,1,1))*mtx3d

    return Wx, W
    
#prunes columns corresponding to gaps in the first sequence. Returns a numpy array
def prunePrimaryGaps(mtx, **kw):
    cutoff = kw.get('cutoff', None)
    cols = np.nonzero(mtx[0] - 20)[0]
    mtxP = mtx[:,cols]
    M,L  = np.shape(mtxP)
    ats  = np.arange(L, dtype=int) + 1
    if cutoff is not None:
        H = marginal_entropy(mtxP)
        cols =  np.where(H > cutoff)[0]
        mtxP = mtxP[:,cols]
        ats  = ats[cols]
    return ats,mtxP


    return Wx,W
                 
def project_aln(aln,Wx,W):
    """
    Calculation of 2D weight matrix.

    Helper function of Rama Ranganathan MATLAB sca5 function. 
    """
    nseq,npos,naa = Wx.shape
    f = getModesFreqs2D(aln)
    
    p_wf = np.zeros((npos,naa))
    wf = np.zeros((naa,npos))
    
    for i in range(npos):
        for j in range(naa):
            wf[j,i] = W[i,j]*f[i,j]
        if np.linalg.norm(wf[:,i])>0: 
            p_wf[i,:] = wf[:,i]/np.linalg.norm(wf[:,i])

    pwX_wf = np.zeros((nseq,npos))
    
    for i in range(npos):
        for j in range(naa):
            pwX_wf[:,i] = pwX_wf[:,i]+p_wf[i,j]*Wx[:,i,j]

    return pwX_wf,p_wf        

# sca5.m
def sca5(mtx, **kwargs):
    """
    Calculates evolutionary covariance matrix according to 
    Rama Ranganathan MATLAB sca5 function, using the relative
    entropy of each sequence and position.

    Returns position covariance and sequence covariance, as well as 
    raw weight matrix.
    """
    bgq = kwargs.get('bgq', estimateBGFreq(mtx))
    nseq,npos = mtx.shape

    mtx3d = binMatrix3D(mtx)
    Wx,W = weightMatrix(mtx3d,bgq)

    pwX,pm = project_aln(mtx,Wx,W)
    
    pwX = np.matrix(pwX)
    Cp = np.abs((pwX.T*pwX)/nseq-np.mean(pwX,axis=0).T*np.mean(pwX,axis=0))
    Cs = np.abs((pwX*pwX.T)/npos-np.mean(pwX.T,axis=0).T*np.mean(pwX.T,axis=0))

    return Cp,Cs,pwX

# return frequency for all aa in  all positions
def getModesFreqs2D(mtx):
    P = np.shape(mtx)[1]
    f2d = np.zeros((P,20))

    for i in range(P):
        bins = np.bincount(mtx[:,i])
        for j in range(np.min([bins.size,20])):
            f2d[i,j] = float(bins[j])/np.sum(bins)
    return f2d

# estimates background frequency of amino acids
def estimateBGFreq(mtx):
    return np.bincount(mtx.flatten(), None, 21)/float(len(mtx.flatten()))

def prune_gaps(mtx, cutoff=None):
    if cutoff is None:
        cutoff = 0.
    m,l = mtx.shape
    ats,mtx = prunePrimaryGaps(mtx, cutoff=cutoff)
    modes = np.array(map(np.argmax, map(np.bincount, mtx.T)))
    idx = np.where(modes != 20)[0]
    return ats[idx], mtx[:,idx]

# Given fasta filename, collect heads and their sequences
def importFasta(fastaFN):
    lines = open(fastaFN, 'r').readlines()
    headers = []
    seq = []
    # Check for .free or .clustal filetype.o
    # No idea what .free format is -- just going off of what comes in the Input
    # folder of sca5 files.
    if lines[0][0] == '>':
        for line in lines:

            if line[0] == '>':
                headers.append(line)
                seq.append('')
            else:
                seq[-1] = seq[-1] + re.sub(r'[^-gascvtpildneqmkhfyrwGASCVTPILDNEQMKHFYRW]', '', line)
    elif lines[0][0].isdigit():
        for line in lines:
            splitline = line.split()
            headers.append(splitline[0])
            seq.append('')
            seq[-1] = seq[-1] + splitline[1]
    else: 
        # Other free possibility: last 
        for line in lines:
            splitline = line.split()
            seq.append('')
            seq[-1] = seq[-1] + splitline[0]
    return headers, seq

#Let the information theoretic functions live down here:
def marginal_probabilities(mtx, **kw):
    m,l = mtx.shape
    W = kw.get('weights', np.ones(m)/float(m))
    return np.vstack([np.matmul(W, mtx==i) for i in range(21)])

def marginal_entropy(mtx, **kw):
    m,l = mtx.shape
    W = kw.get('weights', np.ones(m)/float(m))
    p = marginal_probabilities(mtx, weights=W)
    return -np.sum(p*np.log2(p + (p==0.)), 0)

def joint_entropy(mtx, **kw):
    m,l = mtx.shape
    k = mtx.max() + 1
    W = kw.get('weights', np.ones(m)/float(m))
    J = np.zeros((l,l))
    for i in range(k):
        for j in range(k):
            p = np.matmul(W*(mtx==i).T, mtx==j)
            J = J - p*np.log2(p + (p==0))
    return J

def chi2(mtx, **kw):
    m,l = mtx.shape
    k = mtx.max() + 1
    W = kw.get('weights', np.ones(m)/float(m))
    J = np.zeros((l,l))
    for i in range(k):
        for j in range(k):
            e = np.outer(np.matmul(W,(mtx==i)), np.matmul(W,(mtx==j)))
            x,y = np.nonzero(e)
            o = np.matmul(W*(mtx==i).T, mtx==j)
            J[x,y] = J[x,y] + np.square(o[x,y]-e[x,y])/e[x,y]
    return J

def mutual_information(mtx, **kw):
    m,l = mtx.shape
    W = kw.get('weights', np.ones(m)/float(m))
    H = marginal_entropy(mtx, weights=W)
    J = joint_entropy(mtx, weights=W)
    return H[:,None] + H - J

def pointwise_mutual_information(mtx, **kw):
    m,l = mtx.shape
    k = mtx.max() + 1
    W = kw.get('weights', np.ones(m)/float(m))
    MP= marginal_probabilities(mtx, weights=W)
    PMI = np.zeros((l,l))
    for i in range(k):
        for j in range(k):
            p = np.matmul(W*(mtx==i).T, mtx==j)
            p = p/np.outer(MP[i], MP[j])
            p[np.isnan(p)] = 0.
            PMI = PMI + np.log2(p + (p==0))
    return PMI

def apcnmi(mtx, **kw):
    m,l = mtx.shape
    W = kw.get('weights', np.ones(m)/float(m))
    H = marginal_entropy(mtx, weights=W)
    J = joint_entropy(mtx, weights=W)
    I = H[:,None] + H - J
    return (I - np.outer(np.mean(I, 0), np.mean(I, 0))/np.mean(I))/J

def nmi(mtx, **kw):
    m,l = mtx.shape
    W = kw.get('weights', np.ones(m)/float(m))
    J = joint_entropy(mtx, weights=W)
    H = np.diag(J)
    return (H[:,None] + H - J)/J

def apcmi(mtx, **kw):
    m,l = mtx.shape
    W = kw.get('weights', np.ones(m)/float(m))
    I = mutual_information(mtx, weights=W)
    return I - np.outer(np.mean(I, 0), np.mean(I, 0))

def redundancy(mtx, **kw):
    m,l = mtx.shape
    W = kw.get('weights', np.ones(m)/float(m))
    I = mutual_information(mtx, weights=W)
    H = marginal_entropy(mtx, weights=W)
    return I/(H[:,None] + H)

def kl_div_over_bg(mtx, **kw):
    m,l = mtx.shape
    weights    = kw.get('weights', np.ones(m)/float(m))
    p = marginal_probabilities(mtx, weights = weights)
    background = kw.get('background', np.mean(p,1))
    quot = p/background[:,None]
    quot[quot > 0.] = np.log(quot[quot > 0.])
    return np.sum(p*quot)


def filter_align(s, cover_thresh=0.75, gap_thresh=0.25, entropy_cutoff=0.25):
    ats,mtx = prunePrimaryGaps(binMatrix(s))
    m,l = mtx.shape
    s = [j for i,j in enumerate(s) if np.sum(mtx[i] != 20) > cover_thresh*l]
    ats,mtx = prunePrimaryGaps(binMatrix(s), cutoff=entropy_cutoff)
    m,l = mtx.shape
    ats = ats[np.sum(mtx == 20, 0) < gap_thresh*m]
    mtx = mtx[:,np.sum(mtx == 20, 0) < gap_thresh*m]
    return ats,mtx
    
