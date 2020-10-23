####Useful functions####
import numpy as np
from functools import reduce


si = np.array([[1., 0.],
               [0., 1.]])

sx = np.array([[0., 1.],
               [1., 0.]])

sy = np.array([[0., -1.j],
               [1.j, 0.]]) 

sz = np.array([[1., 0.],    
               [0., -1.]])

spin = [np.array([1., 0.]), np.array([0., 1.])]



def Nbit_single(N):
    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensorl(op_list))

        op_list[n] = sy
        sy_list.append(tensorl(op_list))

        op_list[n] = sz
        sz_list.append(tensorl(op_list))

    return sx_list, sy_list, sz_list



def tensorl(ml):
    return reduce(np.kron, ml, 1)



def get_basis(ind, N):
    basis =  [int(i) for i in bin(ind)[2:]]
    for i in range(N-len(basis)):
        basis.insert(0, 0) 
    return np.array(basis)

def get_ind(basis):
    biconv = 2**np.arange(len(basis))
    ind = np.dot(basis, biconv[::-1].T)
    return ind


def mat_psi(psi, bitA):
    lenrow = 2**(len(bitA))
    lencol = len(psi)/lenrow
    N = int(np.log2(len(psi)))
    mat = np.zeros((int(lenrow), int(lencol)))
    
    for i in range(len(psi)):
        basis = get_basis(i, N)
        basisA = basis[bitA]
       
        bitB = [i for i in range(len(basis))  if i not in bitA]
        basisB = basis[bitB]

        indA = get_ind(basisA)
        indB = get_ind(basisB)
        mat[indA, indB] = psi[i]
    
    return mat

def measure_nncorr(N, prob, baseobs):
    expect = 0
    for i in range(N-1):
        bitA= [i, i+1]
        mat_prob = mat_psi(prob, bitA)
        reduce_prob = np.sum(mat_prob, axis=1)
        expect += baseobs.dot(reduce_prob)
    
    return expect

def measure_singlebit(N, prob, baseobs):
    expect = 0
    for i in range(N):
        bitA = [i]
        mat_prob = mat_psi(prob, bitA)
        reduce_prob = np.sum(mat_prob, axis=1)
        expect += baseobs.dot(reduce_prob)

    return expect
    

def get_baselocal(n):
    NA = n
    basisN = int(2**NA)
    baseobs = np.zeros(basisN)
    for i in range(basisN):
        basisA = get_basis(i, NA)
        baseobs[i] = (-1)**(np.sum(basisA))

    return baseobs


def get_baseglobalz(N):
    basisN = int(2**N)
    baseobs = np.zeros(basisN)
    for i in range(basisN):
        basis = get_basis(i, N)
        baseobs[i] = np.sum([(-1)**(basis[j]) for j in range(N)])

    return baseobs

def get_baseglobalzz(N):
    basisN = int(2**N)
    baseobs = np.zeros(basisN)
    for i in range(basisN):
        basis = get_basis(i, N)
        baseobs[i] = np.sum([(-1)**(basis[j]+basis[j+1]) for j in range(N-1)])

    return baseobs

def get_baseH(H, N):
    ###using matrix operation
    basisN = int(2**N)
    basispsi = np.zeros(basisN)

    basisH = []
    for i in range(basisN):
        basispsi[i] = 1.0
        basisH.append(np.real(np.dot(np.conj(basispsi).T, np.dot(H, basispsi))))
        basispsi[i] = 0
        
    return np.array(basisH)


def gen_btsl(N):
    # btsl = [[0], [1]]
    # for i in range(N-1):
    #     btsn = []
    #     for bts in btsl:
    #         btsn.append(bts + [1])
    #         bts += [0]
    #     btsl.extend(btsn)

    # btsl = np.array(btsl)

    basisN = int(2**N)
    btsl = [get_basis(ind, N) for ind in range(basisN)]
    return btsl


def get_sym_basis(basis):
    #return basis ind with total up spin number 0, 1, 2, 3,..., N
    N = len(basis[0])
    basisblock = [[] for i in range(N + 1)]
    basisblockind = [[] for i in range(N + 1)]
    for bi in basis:
        n = int(np.sum(bi))
        basisblock[n].append(bi.tolist())
        basisblockind[n].append(get_ind(bi))

    return basisblock, basisblockind

def get_block(obj, n, basisblockind):
    blockind = basisblockind[int(n)]
    if len(obj.shape) == 1:
        return obj[blockind]
    elif len(obj.shape) == 2:
        return obj[blockind, :][:, blockind]


def arb_twoXXgate(i,j,N):
    oplist = [si for n in range(N)]
    oplist[i], oplist[j] = sx, sx
    return tensorl(oplist)

def arb_twoYYgate(i,j,N):
    oplist = [si for n in range(N)]
    oplist[i], oplist[j] = sy, sy
    return tensorl(oplist)

def arb_twoZZgate(i,j,N):
    oplist = [si for n in range(N)]
    oplist[i], oplist[j] = sz, sz
    return tensorl(oplist)