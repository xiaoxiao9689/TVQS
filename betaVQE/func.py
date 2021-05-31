
import numpy as np
from error_gate import *
from util import *
import scipy.optimize
from functools import partial
#from scipydirect import minimize
import os
# from zoopt import Dimension, Objective, Parameter, Opt, Solution, ExpOpt
#from gpso import minimize
#from generateRandomProb import generateRandomProb_s
import config
from multiprocessing import Pool
#import scipydirect 

#-----------two bit XY gate error parameter------------------#
Len_theta = 0.05
Len_phi = 0.05


ERROR_XY_LIST = []
for i in range(6):
    Dtheta = (np.random.rand() - 0.5)/0.5*Len_theta
    Dphim = np.sqrt(Len_phi**2 - (Len_phi*Dtheta)**2/Len_theta**2)
    Dphi = (np.random.rand()-0.5)/0.5*Dphim
    ERROR_XY_LIST.append((Dtheta, Dphi))

# print("ERROR_XY_LIST: ", ERROR_XY_LIST)


#--------------------Global constant-------------------------#
N = config.N
angle = config.angle
layer_number = config.layer_number
Id = tensorl([si for i in range(N)])
Totbasis = gen_btsl(N)


#--------------variable and func for experimental spin system--------------#
w_workc = 4930.885 * 2 * np.pi
w_idle = np.array([5434.048, 4930.885, 5376.909, 4974.933, 5473.404]) * 2 * np.pi - w_workc  #MHz 
w_work = (np.array([4930.885]*N) 
#- np.array([0.8, 0, 1.8, 3.2, 2.3])
)* 2 * np.pi - w_workc #MHz 
gn = np.array([14.192, 14.271, 14.283, 14.2]) * 2 * np.pi  #MHz
gnn = np.array([1.142, 0.682, 1.207]) * 2 * np.pi 

tauz = 0.025 #us
taug = 0.009 #us
g = 14.4 * 2 * np.pi

Sx, Sy, Sz = Nbit_single(N)
Ising = 0.0  #experimental frame
for i in range(N-1):
    Ising += gn[i] * (Sx[i].dot(Sx[i+1]))

#NNN term
for i in np.arange(N-2):
    Ising += gnn[i] * (Sx[i].dot(Sx[i+2]))

XY = 0 # Work point frame
for i in range(N-1):
    XY += gn[i]/2 * (Sx[i].dot(Sx[i+1]) + Sy[i].dot(Sy[i+1]))
    XY += 0.5 * g * Sz[i].dot(Sz[i+1])
# for i in range(N-2):
#     XY += gnn[i]/2 * (Sx[i].dot(Sx[i+2]) + Sy[i].dot(Sy[i+2]))

Hint = XY


#exact gate
XYp = 0
for i in range(N-1):
    XYp += Sx[i].dot(Sx[i+1]) + Sy[i].dot(Sy[i+1])

# XYz = 0
# for i in range(N-1):
#     XYz += Sz[i].dot(Sz[i+1])

ChainXY = expm(-1j * angle * XYp) 


def rz_layer(thetal, tau=tauz):
    #experimental frame
    w_theta = w_idle + thetal/tauz
    # print("spin theta:", w_theta)
    Hsingle = Hint.copy()
    for i in range(N):
        Hsingle += -w_theta[i]/2 * Sz[i]
    return  expm(-1j * tau * Hsingle)


def int_layer(tau=taug):
    #experimental frame
    # print("spin work: ", w_work)
    Hglobal = Hint.copy()
    for i in range(N):
        Hglobal += w_work[i]/2 * Sz[i]
    U = expm(-1j * tau * Hglobal)
    return U

print("spin idle: ", w_idle)
def U_idle(t_idle):
    #transform to idle_point
    H = 0.0
    for i in range(N):
        H += -w_idle[i]/2 * Sz[i]
    return expm(1j * t_idle * H)

def U_wkpoint(t_work):
    #transform to work point
    H = 0.0
    for i in range(N):
        H += w_work[i]/2 * Sz[i]
    return expm(1j * t_work * H)

tau_idle = (tauz + taug) * layer_number #+ taug
Int_layer = int_layer()
U_IDLE = U_idle(tau_idle)


BasisBlock, BasisBlockInd = get_sym_basis(Totbasis)
Basez = get_baseglobalz(N)
Basezz = get_baseglobalzz(N)

Baselocalz = get_baselocal(1)
Baselocalzz = get_baselocal(2)


##-------variables and func for 2-state experimental simulation------
ndarray = False
def get_basisn(x, N, n=2):
    basis = []
    while True:
        s, y = divmod(x, n)
        basis.append(y)
        if s == 0:
            break
        x = s
    basis.reverse()
    for i in range(N - len(basis)):
        basis.insert(0, 0)
    return np.array(basis)

def get_indn(basis, n):
    conv = n**np.arange(len(basis))
    ind = np.dot(basis, conv[::-1].T)
    return ind

def Nsite_op(N, i, op):
    oplist = [qtp.qeye(3) for i in range(N)]
    oplist[i] = op
    return qtp.tensor(oplist)

def get_psi(forknums):
    psi = qtp.tensor([Boson[int(i)] for i in forknums])
    return psi

def get_baseallz(N, n):
    basisN = int(n**N)
    baseobs = []
    for i in range(basisN):
        basis = get_basisn(i, N, n=n)
        if not (2 in basis):
            baseobs.append(np.sum([(-1)**(basis[j]) for j in range(N)]))
    return np.array(baseobs)

def get_baseallzz(N, n):
    basisN = int(n**N)
    baseobs = []
    for i in range(basisN):
        basis = get_basisn(i, N, n=n)
        if not (2 in basis):
            baseobs.append(np.sum([(-1)**(basis[j]+basis[j+1]) for j in range(N-1)]))
    return np.array(baseobs)

baseZ3 = get_baseallz(N, 3)
baseZZ3 = get_baseallzz(N, 3)

def cut_psi(psi):
    c2 = []
    for i in range(len(psi)):
        basis = get_basisn(i, N, n=3)
        if not (2 in basis):
            c = psi[i]
            c2.append(c * np.conj(c))

    return np.array(c2)/np.sum(c2)


def siteN(i, w_01):
    w_12 = w_01 + Delta[i]
    siten = qtp.qdiags([[0, w_01, w_01 + w_12]], [0])
    return Nsite_op(5, i, siten)

adag = qtp.create(3)
a = adag.dag()
Adag = [Nsite_op(N, i, adag) for i in range(5)]
A = [Nsite_op(N, i, a) for i in range(5)]

Boson = [qtp.basis(3, i) for i in range(3)]

Delta = -np.array([250., 200., 250., 200., 250.]) * 2 * np.pi 
W_01_idle = w_idle - w_work

##Cutted Hilbert space
sx_c = Boson[0] * Boson[1].dag() + Boson[1] * Boson[0].dag()
sy_c = -1j * Boson[0] * Boson[1].dag() + 1j * Boson[1] * Boson[0].dag()
sz_c = Boson[1] * Boson[1].dag() #+ 2.* Boson[2] *Boson[2].dag()

Sx_c = [Nsite_op(N, i, sx_c) for i in range(N)]
Sy_c = [Nsite_op(N, i, sy_c) for i in range(N)]
Sz_c = [Nsite_op(N, i, sz_c) for i in range(N)]

ry_pid2 = ((-1j*sy_c*np.pi/2) / 2).expm()
Ry_pid2 = qtp.tensor([ry_pid2]*N)

rx_pid2 = ((-1j*sx_c*np.pi/2) / 2).expm()
Rx_pid2 =  (qtp.tensor([rx_pid2]*N))
if ndarray:
    Rx_pid2 = Rx_pid2.data.toarray()
    Ry_pid2 = Ry_pid2.data.toarray()

Hint2 = 0.0
for i in range(4):
    Hint2 += gn[i] * (Adag[i] * A[i+1] + Adag[i+1]*A[i])

for i in range(3):
    Hint2 += gnn[i] * (Adag[i] * A[i+2] + Adag[i+2]*A[i])

#Work point

def int_layer2(tau=taug): 
    w_g = (w_idle[1]-w_work)
    #print("boson work", w_g)
    Hglobal = Hint2.copy()
    for i in range(5):
        Hglobal += siteN(i, w_g[i])
    U = (-1j * tau * Hglobal).expm()
    return U

#rz point
def rz_layer2(thetal, tau=tauz):
    w_theta = w_idle + thetal/tauz - w_work
    #print("boson theta", w_theta)
    Hrz = Hint2.copy()
    for i in range(5):
        Hrz += siteN(i, w_theta[i])
    U = (-1j * tau * Hrz).expm()
    if ndarray:
        U = U.data.toarray()
    return U

print("boson idle: ", W_01_idle)
def U_idle2(t_idle):
    Hidle = 0.0
    for i in range(N):
        Hidle += siteN(i, W_01_idle[i])
    return (1j* t_idle * Hidle).expm()


Int_layer2 = int_layer2(tau=taug)
U_IDLE2 = U_idle2(tau_idle)

# # #Conver the variable to ndarray
if ndarray:
    Int_layer2 = Int_layer2.data.toarray()
    U_IDLE2 = U_IDLE2.data.toarray()


##-------------variables and func for decoherence-------------------
#g = 16 * 2 * np.pi #MHz
tau = angle / g
T1 = 30 #us
T2 = 5 #us
Hxy = qtp.Qobj(XY, dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper', isherm = True)
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

#Uprop = qtp.propagator(Hxy, tau, C_ops)

# FCXY = np.zeros((2 ** N, 2 ** N), dtype = np.complex128)
# for i in range(N - 1):
#     for j in range(i + 1, N):
#         FCXY += arb_twoXXgate(i, j, N)+arb_twoYYgate(i, j, N)
# UFCXY = expm(-1j * np.pi/8 * FCXY)




class Hamiltonian():
    def __init__(self, h = 0.1, N = 5, delta=0.5, Htype = "XY"):
        self.h = h
        self.N = N
        self.delta = delta
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
                self.H += self.delta * np.dot(sz_list[i], sz_list[i + 1])

            for i in range(self.N):
                self.H +=  -self.h * sz_list[i]


def jointprob(btsl, phi):
    prob = []
    for x in btsl:
        singprob = phi ** (x) * (1 - phi) ** (1 - x)
        prob.append(reduce(lambda a, b: a * b, singprob, 1))

    prob = np.array(prob)
    return prob


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

def thermal_ave(op, beta, e, ev):
    pe = np.exp(-beta*e)/np.sum(np.exp(-beta*e))

    obl = np.zeros(len(e))
    for i in range(len(e)):
        vec = ev[:, i]
        obl[i] = np.dot(np.conj(vec).T.dot(op), vec)
    
    return np.dot(pe, obl)

def corr_exact(beta, H):
    eig, eigvec = np.linalg.eigh(H)
    corrl = [thermal_ave(Sz[0].dot(Sz[i]), beta, eig, eigvec) for i in range(1, N)]
    return corrl

def chain_XY(theta, N, qN = None):
    #xy = XY
    chainXY = ChainXY
    if not (qN == None):
        #xy = get_block(XY, qN, BasisBlockInd)
        chainXY = get_block(chainXY, qN, BasisBlockInd)

    return chainXY


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

def singleZlayer(theta):
    return singlelayer(theta, N, "Z")




def Ncnot_nn(c, t, N):
    oplist = [si for i in range(N-1)]
    ind = min(c, t)
    oplist[ind] = cnot(c-ind)
    return tensorl(oplist)




#   # ##-----------Not used for 2-bit gate verison---------
#     # #----------------------------------------------------
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

# def sq_gate(para, noise=False):
#     return np.dot(rz(para[2], noise=noise), np.dot(ry(para[1], noise=noise), rz(para[0], noise=noise)))


# def enhan_XYgate(para, noise=False, dthedphi=None):
#     en_XY = np.kron(rz(para[0], noise=noise), rz(para[1], noise=noise))
#     xy = rxy(np.pi / 8, noise=noise, dthedphi=dthedphi)
#     en_XY = np.dot(xy, en_XY)
#     en_XY = np.dot(np.kron(rz(para[3], noise=noise), si), en_XY)
    
#     return en_XY


# def var_iswap(theta):
#     U = np.kron(rz(-np.pi / 4, False), rz(np.pi / 4, False))
#     U = np.dot(sqrt_iswap(), U)
#     rztheta = np.kron(rz(theta[0] + np.pi, False), rz(-theta[1], False))
#     U = np.dot(rztheta, U)
#     U = np.dot(sqrt_iswap(), U)
#     rztheta = np.kron(rz(5 * np.pi /4, False), rz(-np.pi /4, False))
#     U = np.dot(rztheta, U)
#     return(U)


# def XY_layer(para, N, noise, qN = None, iswap = True, dthedphi=None):
#     U = 1.0
#     op_list = [si for i in range(N - 1)]
#     for i in range(N - 1):
#         if iswap:
#             theta = (para[int(2 * i + 1)], para[int(2 * i + 2)])
#             op_list[i] = var_iswap(theta)
#             var_iswap_i = tensorl(op_list)
#             U = np.dot(var_iswap_i, U)
#             op_list[i] = si
#         else:
#             op_list[i] = rxy(para[i], noise=noise, dthedphi=dthedphi)
#             xy_i = tensorl(op_list)
#             U = np.dot(xy_i, U)
#             op_list[i] = si
#     return U

# def XY_layer_grad(para, N, noise, a, b, gi, qN=None,  dthedphi=None):
#     U = 1.0
#     op_list = [si for i in range(N - 1)]
#     for i in range(N - 1):
#         if i == gi:
#             op_list[i] = XYgate_grad(para[i], a, b, noise=noise, dthedphi = None)
#         else:
#             op_list[i] = rxy(para[i], noise=noise, dthedphi=dthedphi)
#         xy_i = tensorl(op_list)
#         U = np.dot(xy_i, U)
#         op_list[i] = si
    
#     return U



# # def fc_XY(theta, N, qN = None):
# #     #fcxy = FCXY
# #     ufcxy = UFCXY
# #     if not (qN == None):
# #         #fcxy = get_block(FCXY, qN, BasisBlockInd)
# #         ufcxy = get_block(ufcxy, qN, BasisBlockInd)
# #     return  ufcxy


# def XYgate_grad(para, a, b, noise=False, dthedphi=None):
#     # XY_grad = np.kron(rz(para[0], noise), rz(para[1], noise))
#     # xy = rxy(para[2]+a, noise, dthedphi=dthedphi) 
#     # XY_grad = np.dot(xy, XY_grad)
#     # XY_grad = np.dot(np.kron(sx, si), XY_grad)
#     # XY_grad = np.dot(rxy(b, noise, dthedphi=dthedphi), XY_grad)
#     # XY_grad = np.dot(np.kron(sx, si), XY_grad)
#     # XY_grad = np.dot(np.kron(rz(para[3], noise), si), XY_grad)

#     XY_grad = rxy(para + a, noise=False, dthedphi=dthedphi)
#     XY_grad = np.dot(np.kron(sx, si), XY_grad)
#     XY_grad = np.dot(rxy(b, noise=False, dthedphi=dthedphi), XY_grad)
#     XY_grad = np.dot(np.kron(sx, si), XY_grad)

#     return XY_grad