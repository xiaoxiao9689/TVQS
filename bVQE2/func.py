
import numpy as np
from error_gate import *
from util import *
import scipy.optimize
from functools import partial
#from scipydirect import minimize
import os
from zoopt import Dimension, Objective, Parameter, Opt, Solution, ExpOpt
from gpso import minimize
#from generateRandomProb import generateRandomProb_s
import config
from multiprocessing import Pool
import scipydirect 

###two bit XY gate error parameter###
Len_theta = 0.05
Len_phi = 0.05

ERROR_XY_LIST = []
for i in range(6):
    Dtheta = (np.random.rand() - 0.5)/0.5*Len_theta
    Dphim = np.sqrt(Len_phi**2 - (Len_phi*Dtheta)**2/Len_theta**2)
    Dphi = (np.random.rand()-0.5)/0.5*Dphim
    ERROR_XY_LIST.append((Dtheta, Dphi))

# print("ERROR_XY_LIST: ", ERROR_XY_LIST)


#Global constant
N = config.N
Id = tensorl([si for i in range(N)])
Totbasis = gen_btsl(N)

XYlist = [arb_twoXXgate(i, i + 1, N) + arb_twoYYgate(i, i + 1, N) for i in range(N - 1)]
XY = np.sum(XYlist,axis=0)
ChainXY = expm(-1j * np.pi/8 * XY)

g = 16 * 2 * np.pi #MHz
T1 = 30 #us
T2 = 5 #us
Hxy = g * qtp.Qobj(XY, dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper', isherm = True)
C_ops = []
op_list = [qtp.qeye(2) for i in range(N)]
for i in range(N):    
    # op_list[i] = 1 / np.sqrt(T2) * qtp.basis(2, 0) * qtp.basis(2, 0).dag()
    # C_ops.append(qtp.tensor(op_list))
    # op_list[i] = 1 / np.sqrt(T2) * qtp.basis(2, 1) * qtp.basis(2, 1).dag()
    # C_ops.append(qtp.tensor(op_list))
    op_list[i] = 1 / np.sqrt(T2) * qtp.sigmaz()
    C_ops.append(qtp.tensor(op_list))

    op_list[i] = 1 / np.sqrt(T1) * qtp.basis(2, 0) * qtp.basis(2, 1).dag()
    C_ops.append(qtp.tensor(op_list))
    op_list[i] = qtp.qeye(2)


FCXY = np.zeros((2 ** N, 2 ** N), dtype = np.complex128)
for i in range(N - 1):
    for j in range(i + 1, N):
        FCXY += arb_twoXXgate(i, j, N)+arb_twoYYgate(i, j, N)
UFCXY = expm(-1j * np.pi/8 * FCXY)

Sqrt_ISwap = sqrt_iswap() 
BasisBlock, BasisBlockInd = get_sym_basis(Totbasis)

Basez = get_baseglobalz(N)
Basezz = get_baseglobalzz(N)


class Hamiltonian():
    def __init__(self, h = 0.1, N = 5, Htype = 10):
        self.h = h
        self.N = N
        self.H = 0.0
        self.Htype = Htype
        self.set_H()

    def set_H(self):
        sx_list, sy_list, sz_list = Nbit_single(N)
        self.H = 0. + 0.j
        if self.Htype == "XY":
            for i in range(self.N):
                self.H += self.h * sz_list[i]

            for i in range(self.N - 1):
                self.H += np.dot(sx_list[i], sx_list[i + 1]) + np.dot(sy_list[i], sy_list[i + 1])
        
        elif self.Htype == "XXZ":
            for i in range(self.N - 1):
                self.H += np.dot(sx_list[i], sx_list[i + 1]) + np.dot(sy_list[i], sy_list[i + 1])
                self.H += self.h * np.dot(sz_list[i], sz_list[i + 1])

        

def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper

def cnot(control):
    if control == 0:
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]])
    elif control ==1:
        return  np.array([[1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0]])


def jointprob(btsl, phi):
    prob = []
    for x in btsl:
        singprob = phi ** (x) * (1 - phi) ** (1 - x)
        prob.append(reduce(lambda a, b: a * b, singprob, 1))

    prob = np.array(prob)
    return prob


######variational gate########
def sq_gate(para, noise=False):
    return np.dot(rz(para[2], noise=noise), np.dot(ry(para[1], noise=noise), rz(para[0], noise=noise)))


def enhan_XYgate(para, noise=False, dthedphi=None):
    en_XY = np.kron(rz(para[0], noise=noise), rz(para[1], noise=noise))
    xy = rxy(np.pi / 8, noise=noise, dthedphi=dthedphi)
    en_XY = np.dot(xy, en_XY)
    en_XY = np.dot(np.kron(rz(para[3], noise=noise), si), en_XY)
    
    return en_XY


def var_iswap(theta):
    U = np.kron(rz(-np.pi / 4, False), rz(np.pi / 4, False))
    U = np.dot(sqrt_iswap(), U)
    rztheta = np.kron(rz(theta[0] + np.pi, False), rz(-theta[1], False))
    U = np.dot(rztheta, U)
    U = np.dot(sqrt_iswap(), U)
    rztheta = np.kron(rz(5 * np.pi /4, False), rz(-np.pi /4, False))
    U = np.dot(rztheta, U)
    return(U)



def singlelayer(para, N, singlegate, noise=False, qN=None):
    gatel = []
    if singlegate == "X":
        gatel = [rx(para[i], noise=noise) for i in range(N)]
    elif singlegate == "Y":
        gatel = [ry(para[i], noise=noise) for i in range(N)]
    elif singlegate == "Z":
        if qN == None:
            gatel = [rz(para[i], noise=noise) for i in range(N)]
        else:
            gatel = [qN_rz(para[i], noise=noise) for i in range(N)]

    singlelayer = tensorl(gatel)
    if not (qN == None):  
        singlelayer = get_block(singlelayer, qN, BasisBlockInd)
    return singlelayer  


def XY_layer(para, N, noise, qN = None, iswap = True, dthedphi=None):
    U = 1.0
    op_list = [si for i in range(N - 1)]
    for i in range(N - 1):
        if iswap:
            theta = (para[int(2 * i + 1)], para[int(2 * i + 2)])
            op_list[i] = var_iswap(theta)
            var_iswap_i = tensorl(op_list)
            U = np.dot(var_iswap_i, U)
            op_list[i] = si
        else:
            op_list[i] = rxy(para[i], noise=noise, dthedphi=dthedphi)
            xy_i = tensorl(op_list)
            U = np.dot(xy_i, U)
            op_list[i] = si
    return U

def XY_layer_grad(para, N, noise, a, b, gi, qN=None,  dthedphi=None):
    U = 1.0
    op_list = [si for i in range(N - 1)]
    for i in range(N - 1):
        if i == gi:
            op_list[i] = XYgate_grad(para[i], a, b, noise=noise, dthedphi = None)
        else:
            op_list[i] = rxy(para[i], noise=noise, dthedphi=dthedphi)
        xy_i = tensorl(op_list)
        U = np.dot(xy_i, U)
        op_list[i] = si
    
    return U

def chain_XY(theta, N, qN = None):
    #xy = XY
    chainXY = ChainXY
    if not (qN == None):
        #xy = get_block(XY, qN, BasisBlockInd)
        chainXY = get_block(chainXY, qN, BasisBlockInd)

    return chainXY

def fc_XY(theta, N, qN = None):
    #fcxy = FCXY
    ufcxy = UFCXY
    if not (qN == None):
        #fcxy = get_block(FCXY, qN, BasisBlockInd)
        ufcxy = get_block(ufcxy, qN, BasisBlockInd)
    return  ufcxy


def XYgate_grad(para, a, b, noise=False, dthedphi=None):
    # XY_grad = np.kron(rz(para[0], noise), rz(para[1], noise))
    # xy = rxy(para[2]+a, noise, dthedphi=dthedphi) 
    # XY_grad = np.dot(xy, XY_grad)
    # XY_grad = np.dot(np.kron(sx, si), XY_grad)
    # XY_grad = np.dot(rxy(b, noise, dthedphi=dthedphi), XY_grad)
    # XY_grad = np.dot(np.kron(sx, si), XY_grad)
    # XY_grad = np.dot(np.kron(rz(para[3], noise), si), XY_grad)

    XY_grad = rxy(para + a, noise=False, dthedphi=dthedphi)
    XY_grad = np.dot(np.kron(sx, si), XY_grad)
    XY_grad = np.dot(rxy(b, noise=False, dthedphi=dthedphi), XY_grad)
    XY_grad = np.dot(np.kron(sx, si), XY_grad)

    return XY_grad
    

def gen_samples(phi, nbatch, N):
    samples = np.zeros((nbatch, N))
    for i in range(nbatch):
        np.random.seed()
        samples[i, :] = phi > np.random.rand(N)
    
    return samples


def logp(phi, x):
    return np.sum(x * np.log(phi) + (1 - x) * np.log(1 - phi))


def exact(beta, H):
    e = np.linalg.eigvalsh(H)
    Z = np.sum(np.exp(-beta*e))
    F = -np.log(Z)/beta
    E = np.sum(np.dot(np.exp(-beta*e),e))/Z
    S = (E-F)*beta

    return F, E, S




