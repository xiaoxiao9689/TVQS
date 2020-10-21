# coding: utf-8

# In[1]:


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import random
#from tqdm import tqdm
import scipy
import scipy.io as sio 
import math
from functools import reduce

def generateRandomProb(probs,stats=3000):
    psize = np.shape(probs)
    single = False
    if len(psize)==1:
        probs = np.asarray([probs])
        psize = np.shape(probs)
        single = True
    lenp1,lenp2 = psize
    pSimus = np.zeros(psize)
    randomPs = np.random.rand(lenp1,stats)
    for idxp in range(lenp1):
        print ('idxp', idxp)
        probs0 = np.hstack([0.0,probs[idxp,:]])
        psLims = np.zeros([lenp2,2])
        for idx in range(lenp2):
            psLims[idx,:] = [np.sum(probs0[0:idx+1]),np.sum(probs0[0:idx+2])]
        for idx0 in range(lenp2):
            pSimus[idxp,idx0] = sum((randomPs[idxp,:]>psLims[idx0,0]) & (randomPs[idxp,:]<=psLims[idx0,1]))/float(stats)
    if single:
        pSimus = pSimus[0]
    return pSimus
    
def getFidMatrix(fids,returnInv=True):
    fidMatrix = np.array([[1]])

    for i in range(int(len(fids)/2)):
        f0 = fids[2*i]
        f1 = fids[2*i+1]
        mat = np.array([[f0,1-f1],[1-f0,f1]])
        fidMatrix = np.kron(fidMatrix,mat)
    if returnInv:
        fidMatrixInv = np.linalg.inv(fidMatrix)
        return fidMatrixInv 
    else:
        return fidMatrix    
measurefids = np.array([0.97, 0.91] *10)  ### Here, the actual fidelities of 10 qubits during the measurement need to be used.
