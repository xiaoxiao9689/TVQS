
import numpy as np
from error_gate import *
from util import *
from scipy.linalg import expm
import scipy.optimize
from functools import partial
#from scipydirect import minimize
import os
from zoopt import Dimension, Objective, Parameter, Opt, Solution, ExpOpt
from gpso import minimize
from  generateRandomProb import generateRandomProb_s
import config
import multiprocessing
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

print("ERROR_XY_LIST: ", ERROR_XY_LIST)


#Global constant
N = config.N
Id = tensorl([si for i in range(N)])
Totbasis = gen_btsl(N)

XYlist = [arb_twoXXgate(i, i + 1, N) + arb_twoYYgate(i, i + 1, N) for i in range(N - 1)]
XY = np.sum(XYlist,axis=0)

g = 16 * 2 * np.pi #MHz
T1 = 30 #us
T2 = 5 #us
Hxy = g * qtp.Qobj(XY, dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper', isherm = True)
C_ops = []
op_list = [qtp.qeye(2) for i in range(N)]
for i in range(N):    
    op_list[i] = 1 / np.sqrt(T2) * qtp.basis(2, 0) * qtp.basis(2, 0).dag()
    C_ops.append(qtp.tensor(op_list))
    op_list[i] = 1 / np.sqrt(T2) * qtp.basis(2, 1) * qtp.basis(2, 1).dag()
    C_ops.append(qtp.tensor(op_list))
    op_list[i] = 1 / np.sqrt(T1) * qtp.basis(2, 0) * qtp.basis(2, 1).dag()
    C_ops.append(qtp.tensor(op_list))


FCXY = np.zeros((2 ** N, 2 ** N), dtype = np.complex128)
for i in range(N - 1):
    for j in range(i + 1, N):
        FCXY += arb_twoXXgate(i, j, N)+arb_twoYYgate(i, j, N)

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
def sq_gate(para, noise):
    return np.dot(rz(para[2], noise), np.dot(ry(para[1], noise), rz(para[0], noise)))

            
def enhan_XYgate(para, noise, dthedphi=None):
    en_XY = np.kron(rz(para[0], noise), rz(para[1], noise))
    xy = rxy(para[2], noise, dthedphi=dthedphi)
    en_XY = np.dot(xy, en_XY)
    en_XY = np.dot(np.kron(rz(para[3], noise), si), en_XY)
    
    return en_XY

def singlelayer(para, N, singlegate, noise, qN = None):
    gatel = []
    if singlegate == "X":
        gatel = [rx(para[i], noise) for i in range(N)]
    elif singlegate == "Y":
        gatel = [ry(para[i], noise) for i in range(N)]
    elif singlegate == "Z":
        if qN == None:
            gatel = [rz(para[i], noise) for i in range(N)]
        else:
            gatel = [qN_rz(para[i], noise) for i in range(N)]

    singlelayer = tensorl(gatel)
    if not (qN == None):  
        singlelayer = get_block(singlelayer, qN, BasisBlockInd)
    return singlelayer  

def chain_XY(theta, N, noise, qN = None):
    xy = XY
    if not (qN == None):
        xy = get_block(XY, qN, BasisBlockInd)
    
    return expm(-1j * theta * xy)

def fc_XY(theta, N, noise, qN = None):
    fcxy = FCXY
    if not (qN == None):
        fcxy = get_block(FCXY, qN, BasisBlockInd)

    return  expm(-1j * theta * fcxy)


def XYgate_grad(para, a, b, noise, dthedphi=None):
    XY_grad = np.kron(rz(para[0], noise), rz(para[1], noise))
    xy = rxy(para[2]+a, noise, dthedphi=dthedphi) 
    XY_grad = np.dot(xy, XY_grad)
    XY_grad = np.dot(np.kron(sx, si), XY_grad)
    XY_grad = np.dot(rxy(b, noise, dthedphi=dthedphi), XY_grad)
    XY_grad = np.dot(np.kron(sx, si), XY_grad)
    XY_grad = np.dot(np.kron(rz(para[3], noise), si), XY_grad)

    return XY_grad


def get_U_transymm(thetal, layer_number, N, gate, Noise, qN = None):
    '''Employ translation symmetry in the bulk of the system '''

    if gate == "XY":
        pass


def get_U(thetal, layer_number, N, gate, noise, qN = None):
    if gate == "XY":
        theta = thetal.reshape((layer_number, N - 1, 4))
        U = 1.0
        for l in range(theta.shape[0]): # layer
            Ul = 1.0
            for i in range(theta.shape[1]):
                oplist = [si for n in range(N - 1)] 
                oplist[i] = enhan_XYgate(theta[l, i, :], noise, dthedphi=ERROR_XY_LIST[i])
                tq_gateN = tensorl(oplist)
                Ul = np.dot(tq_gateN , Ul)
            U = np.dot(Ul, U)
        
    elif gate == "chainXY":
        theta = thetal.reshape((layer_number, N + 1))
        U = 1.0
        for l in range(theta.shape[0]):
            Ul = singlelayer(theta[l, :N], N, "Z", noise, qN = qN)
            Ul = np.dot(chain_XY(theta[l, N], N, noise, qN = qN), Ul)
            U = np.dot(Ul, U)

    elif gate == "fcXY":
        theta = thetal.reshape((layer_number, N + 1))
        U = 1.0
        for l in range(theta.shape[0]):
            Ul = singlelayer(theta[l, :N], N, "Z", noise, qN = qN)
            Ul = np.dot(fc_XY(theta[l, N], N, noise, qN = qN), Ul)
            U = np.dot(Ul, U)

    return U


def evolve_rho(rho, thetal, layer_number, N, noise):
    theta = thetal.reshape((layer_number, N + 1))
    for l in range(theta.shape[0]):
        sing_layer =  singlelayer(theta[l, :N], N, "Z", noise)
        sing_layer = qtp.Qobj(sing_layer, dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper')
        rho = sing_layer.dag() * rho * sing_layer
        tlist = [0, theta[l, N] / g]
        res = qtp.mesolve(Hxy, rho, tlist, c_ops = C_ops)
        rho = res.states[-1]

    return rho
    

def get_U_grad(thetal, layer_number, N, gate, gi, a, b, noise):
    theta = thetal.reshape((layer_number, N - 1, 4))
    layer_ind = gi // ((N - 1) * 4)
    bit_ind = gi // 4%( N - 1)

    U = Id
    for l in range(theta.shape[0]):
        Ul = Id
        for i in range(theta.shape[1]):
            oplist = [si for n in range(N-1)]
            if (l == layer_ind) and (i == bit_ind):
                oplist[i] = XYgate_grad(theta[l, i, :], a, b, noise, dthedphi=ERROR_XY_LIST[i])
            else:
                oplist[i] = enhan_XYgate(theta[l, i, :], noise,
                dthedphi=ERROR_XY_LIST[i])

            Ul = np.dot(tensorl(oplist), Ul)
        U = np.dot(Ul, U)

    return U


def gen_samples(phi, nbatch, N):
    samples = np.zeros((nbatch, N))
    for i in range(nbatch):
        np.random.seed()
        samples[i, :] = phi > np.random.rand(N)
    
    return samples


def logp(phi, x):
    return np.sum(x * np.log(phi) + (1 - x) * np.log(1 - phi))


def q_expects(Ul, x, H):
    H = H.H
    #Ul is list contain all qN block
    qN = int(np.sum(x))
    U = Ul[qN]
    psi = tensorl([spin[int(i)] for i in x])
    psi = get_block(psi, qN, BasisBlockInd)
    H = get_block(H, qN, BasisBlockInd)
    psi = np.dot(U, psi)

    return np.real(np.dot(np.conj(psi).T, np.dot(H, psi))/np.dot(np.conj(psi).T, psi))


def q_expect_rho(thetal, x, H, layer_number, noise):
    h = H.h
    Htype = H.Htype
    N = H.N
    psi = qtp.tensor([qtp.basis(2, int(i)) for i in x])
    psirho = qtp.ket2dm(psi)
    psirho = evolve_rho(psirho, thetal ,layer_number, N, noise)
    
    # H = qtp.Qobj(H.H,  dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper', isherm = True)
    # Hexpect = np.real(np.trace(H * psirho))

    prob = psirho.diag()
    prob = generateRandomProb_s(prob, stats = 10000)
    ez = np.dot(prob, Basez) / np.sum(prob)
    ezz = np.dot(prob, Basezz) / np.sum(prob)

    gry = qtp.Qobj(tensorl([ry(np.pi / 2, noise) for i in range(N)]), dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper', isherm = True)
    psirhox = gry.dag() * psirho * gry
    prob = psirhox.diag()
    prob = generateRandomProb_s(prob, stats = 10000)
    exx = np.dot(prob, Basezz) / np.sum(prob)

    grx = qtp.Qobj(tensorl([rx(np.pi / 2, noise) for i in range(N)]), dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper', isherm = True)
    psirhoy = grx.dag() * psirho * grx
    prob = psirhoy.diag()
    prob = generateRandomProb_s(prob, stats = 10000)
    eyy = np.dot(prob, Basezz) / np.sum(prob)

    if Htype == "XY":
        Hexpect = -h * ez - exx - eyy
    elif Htype == "XXZ":
        Hexpect = -h * ezz - exx - eyy
    

    return  Hexpect


def q_expect(U, x, H, noise):
    if noise:
        #print("noised")
        h = H.h
        Htype = H.Htype
        psi = tensorl([spin[int(i)] for i in x])
        N = H.N

        psi = np.dot(U, psi)

        prob = np.abs(psi) ** 2
        prob = generateRandomProb_s(prob, stats = 10000)
        ez = np.dot(prob, Basez) / np.sum(prob)
        ezz = np.dot(prob, Basezz) / np.sum(prob)

        psix = np.dot(tensorl([ry(np.pi / 2, noise) for i in range(N)]), psi)
        prob = np.abs(psix) ** 2
        prob = generateRandomProb_s(prob, stats = 10000)
        exx = np.dot(prob, Basezz) / np.sum(prob)

        psiy = np.dot(tensorl([rx(np.pi / 2, noise) for i in range(N)]), psi)
        prob = np.abs(psiy) ** 2
        prob = generateRandomProb_s(prob, stats = 10000)
        eyy = np.dot(prob, Basezz) / np.sum(prob)

        if Htype == "XY":
            Hexpect = -h * ez - exx - eyy
        elif Htype == "XXZ":
            Hexpect = -h * ezz - exx - eyy
        
        return  Hexpect
    
    else:
        H = H.H
        psi = tensorl([spin[int(i)] for i in x])
        psi = np.dot(U, psi)

        return  np.real(np.dot(np.conj(psi).T, np.dot(H, psi))/np.dot(np.conj(psi).T, psi))

   
    ##Ising
    # psi = tensorl([spin[int(i)] for i in x])
    # N = int(np.log2(len(psi)))
    # ##Using total operator####
    # psi = np.dot(U, psi)
    # psix = np.dot(tensorl([ry(np.pi/2, False) for i in range(N)]), psi)
    # prob = np.abs(psix)**2
    # ex = np.dot(prob, Basez)

    # prob = np.abs(psi)**2
    # ezz = np.dot(prob, Basezz)

    # H = -h*ex-ezz

    #return beta*H 


def get_Ex(thetal, samples, H, layer_number, noise = False, gate = "XY", symmetry = False, decoherence = False, parallel = False):
    N = H.N
    core_number = config.core_number
    global wraper

    if symmetry:
        Ul = [get_U(thetal, layer_number, N, gate, noise, qN = n) for n in range(N + 1)]
        if parallel:
            def wraper(args):
                Ul, x, H = args
                return q_expects(Ul, x, H)
            
            pool = multiprocessing.Pool(core_number)      
            parapack = [(Ul, x, H) for x in samples]
            E_x = pool.map(unpack_args(q_expects), parapack)
            E_x = np.array(E_x)
            pool.close()
            pool.join()
        else:
            E_x = np.array([q_expects(Ul, x, H) for x in samples])
    
    elif decoherence:
        if parallel:
            def wraper(args):
                thetal, x, H, layer_number, noise = args
                return q_expect_rho(thetal, x, H, layer_number, noise)  
                   
            pool = multiprocessing.Pool(core_number)      
            parapack = [(thetal, x, H, layer_number, noise) for x in samples]
            E_x = pool.map(wraper, parapack)
            E_x = np.array(E_x)
            pool.close()
            pool.join()
        else:
            E_x = np.array([q_expect_rho(thetal, x, H, layer_number, noise) for x in samples])
        
    else:
        U = get_U(thetal, layer_number, N, gate, noise)
        if parallel:
            def wraper(args):
                U, x, H, noise = args
                return q_expect(U, x, H, noise)  
            parapack = [(U, x, H, noise) for x in samples]
            pool = multiprocessing.Pool(core_number)      
            E_x = pool.map(wraper, parapack)
            E_x = np.array(E_x)
            pool.close()
            pool.join()
        else:
            E_x = np.array([q_expect(U, x, H, noise) for x in samples])
    
    return E_x

def loss_quantum(thetal, alpha, samples, beta, H, layer_number,  noise=False,  gate="XY", samp=False, join=False, symmetry = False, peierls = False, decoherence = False, parallel = False):
    if join:
        phi = np.exp(alpha)/np.sum(np.exp(alpha))
    else:
        phi = 1/(1+np.exp(-alpha))

    E_x = get_Ex(thetal, samples, H, layer_number, noise = noise, gate = gate, symmetry = symmetry,  decoherence = decoherence, parallel = parallel)

    if samp:
        return np.mean(E_x)
    else:
        if join:
            return np.dot(phi, E_x)
        else:
            prob = jointprob(samples, phi)
            if peierls:
                prob = np.exp(-beta * E_x)
                prob = prob/np.sum(prob)
            return np.dot(prob, E_x)


def loss_quantum_grad(thetal, alpha, samples, beta, H, layer_number,  gi, a, b,  noise=False, gate="XY", samp=False, join=False):
    if join:
        phi = np.exp(alpha)/np.sum(np.exp(alpha))
    else:
        phi = 1/(1+np.exp(-alpha))
    
    U = get_U_grad(thetal, layer_number, N, gate, gi, a, b, noise)
    #q_expectl = [q_expect(U, x, H) for x in samples]
    q_expectl = [q_expect(U, x, H, noise) for x in samples]

    if samp: # Using samp
        return np.mean(q_expectl)
    else:
        if join:
            return np.dot(phi, q_expectl)
        else:
            prob = jointprob(samples, phi)
            return np.dot(prob, q_expectl)
    


def loss_func(para, samples, beta, H, layer_number,  nbatch = 100,  noise=False, gate="XY", samp=False, join=False, symmetry = False, decoherence = False, parallel = False):
    N = H.N
    if join:
        Nphi = len(samples)
        phi = np.exp(para[:Nphi])/np.sum(np.exp(para[:Nphi]))
        thetal = para[Nphi:]
        U = get_U(thetal, layer_number, N, gate, noise)
        loss_samp = np.log(phi) + [beta * q_expect(U, x, H, noise) for x in samples]
        return np.dot(phi, loss_samp)

    else:
        phi = 1 / (1 + np.exp(-para[:N]))
        thetal = para[N:]

        E_x = get_Ex(thetal, samples, H, layer_number, noise = noise, gate = gate, symmetry = symmetry,  decoherence = decoherence, parallel = parallel)

        logp_x = np.array([logp(phi, x) for x in samples])
        loss_samp = logp_x + beta * E_x
        
        if samp:
             return  np.mean(loss_samp)
        else:
            prob = jointprob(samples, phi)
            return np.dot(prob, loss_samp) 


def loss_func_bound(para, samples, beta, H, layer_number,  gate="XY", nbatch = 100,  noise=False, samp=False, join=False, symmetry = False, decoherence = False, parallel = False):
    N = H.N
    if join:
        Nphi = len(samples)
        phi = para[:Nphi] 
        thetal = para[Nphi:]
        U = get_U(thetal, layer_number, N, gate, noise)
        loss_samp = np.log(phi) + [beta * q_expect(U, x, H, noise) for x in samples]
        return np.dot(phi, loss_samp)

    else:
        phi = para[:N]
        thetal = para[N:]
        
        if samp:
            samples = gen_samples(phi, nbatch, N)

        #samples = gen_btsl_sub(N, 10)
        E_x = get_Ex(thetal, samples, H, layer_number, noise = noise, gate = gate, symmetry = symmetry,  decoherence = decoherence, parallel = parallel)

        logp_x = np.array([logp(phi, x) for x in samples])
        loss_samp = logp_x + beta * E_x

        if samp:
            return  np.mean(loss_samp)
        else:      
            prob = jointprob(samples, phi)
            return np.dot(prob, loss_samp)

def loss_func_peierls(para, samples, beta, H, layer_number, nbatch = 100,  noise=False, gate="XY", samp=False, join=False, symmetry = False, decoherence = False, parallel = False):
    '''This form of loss can be related to GDM form by take the classical probability p(x) as e^(-E_\theta(x))/Z_theta'''
    thetal = para
    #samples = gen_btsl_sub(N, 10)
    E_x = get_Ex(thetal, samples, H, layer_number, noise = noise, gate = gate, symmetry = symmetry,  decoherence = decoherence, parallel = False)
    
    loss =  -np.log(np.sum(np.exp(-beta * E_x)))
    return loss

##parametershift for grad#######

def grad_parashift(thetal, alpha,  samples, beta,  H, layer_number, gate="XY", samp=False, join=False, noise=False):
    grad_theta = []
    dth = np.zeros(len(thetal))
    for i in range(len(thetal)):
        if not (i % 4 == 2):
            ###parameter shift for single-qubit gate
            dth[i] = np.pi/2
            loss1 = loss_quantum(thetal + dth, alpha, samples, beta, H, layer_number,  gate = gate, samp = samp, join = join, noise = noise)
            dth[i] = -np.pi/2
            loss1 -= loss_quantum(thetal + dth, alpha, samples, beta, H, layer_number,   gate = gate, samp = samp, join=join, noise = noise)
            grad_theta.append(loss1*0.5)
            dth[i] = 0

        else:
            ####parameter shift for XY gate
            lossX = loss_quantum_grad(thetal, alpha, samples, beta, H, layer_number,  i, np.pi/8, np.pi/8,  gate = gate, samp = samp, join = join, noise = noise) - loss_quantum_grad(thetal, alpha, samples, beta, H, layer_number, i, -np.pi / 8, -np.pi / 8, gate = gate, samp = samp, join = join, noise = noise)
            
            lossY = loss_quantum_grad(thetal, alpha, samples, beta, H, layer_number, i, np.pi/8, -np.pi/8,  gate=gate, samp=samp, join=join, noise=noise) - loss_quantum_grad(thetal, alpha, samples, beta, H, layer_number,  i, -np.pi / 8, np.pi / 8, gate = gate, samp = samp, join = join, noise = noise)
            grad_theta.append(lossX +lossY)
        
    return np.array(grad_theta)


def grad(para, samples, beta, H, layer_number,  gate="XY", samp=False, join=False, noise=False, symmetry = False, decoherence = False):
    N = H.N
    #calculate grad by hand
    if join:
        ##total join prob
        alpha = para[:int(2 ** N)]
        phi = np.exp(alpha) / np.sum(np.exp(alpha))
        thetal = para[int(2 ** N):]
        U = get_U(thetal, layer_number, N, gate, noise)
        grad_logp = -np.outer(phi, phi)
        for i in range(len(phi)):
            grad_logp[i, i] = phi[i] - phi[i] ** 2
        
        fx = np.log(phi) + beta * np.array([q_expect(U, x, H, noise) for x in samples])
        grad_phi = (1 / beta) * np.dot((1 + fx), grad_logp)


    else:
        alpha = para[:N]
        phi = 1 / (1 + np.exp(-alpha))
        thetal = para[N:]
        prob = jointprob(samples, phi)
        b = 0.0
        if samp: 
            b = loss_func(para, samples, beta, H, layer_number,  noise = noise,  gate = gate, samp = samp, join = join)
            prob = np.ones(len(samples))/len(samples)
        
        if symmetry:
            Ul = [get_U(thetal, layer_number, N, gate, noise, qN = n) for n in range(N + 1)] 
        else:
            U = get_U(thetal, layer_number, N, gate, noise)

        grad_phil = []
        for x in samples:
            grad_logp = (x - phi)
            if symmetry:
                fx = logp(phi, x) + beta * q_expects(Ul, x, H)
            else:
                fx = logp(phi, x) + beta * q_expect(U, x, H, noise) 
            grad_phil.append((fx - b)*grad_logp)
        grad_phi = np.dot(prob, grad_phil)


    loss_quantum_p = partial(loss_quantum,  gate = gate, samp = samp, join = join, symmetry = symmetry)
    grad_theta = scipy.optimize.approx_fprime(thetal, loss_quantum_p, 1e-8, alpha,  samples, beta,  H, layer_number) 
    #grad_theta = grad_parashift(thetal, alpha, samples, beta, H, layer_number,  gate=gate, samp=samp, join=join, noise=noise)
    # print("difference: ", grad_theta1-grad_theta)           
    grad = np.concatenate((grad_phi, beta * grad_theta))

    ###Diff
    #loss_func_p = partial(loss_func,  nbatch = 100,  noise=False, gate="XY", samp=False, join=False, symmetry = False)
    #loss_func_p = partial(loss_func, nbatch = 100,  noise = noise, gate = gate, samp = samp, join = join, symmetry = symmetry)
    # grad = scipy.optimize.approx_fprime(para, loss_func_p, 1e-8, samples, beta, H, layer_number)

    return grad



def exact(beta, H):
    e = np.linalg.eigvalsh(H)
    Z = np.sum(np.exp(-beta*e))
    F = -np.log(Z)/beta
    E = np.sum(np.dot(np.exp(-beta*e),e))/Z
    S = (E-F)*beta

    return F, E, S



def optimize_scip(niter, layer_number, beta, H, samples, nbatch = 100, gate="XY", samp=False, method='lbfgs', join=False, noise=False):
    np.random.seed()
    N = H.N
    #bounds = [(0, 1)]*N+[(None, None)]*(N-1)*layer_number*15
    if gate == "XY":
        thetal = 2 * np.pi * np.random.rand(layer_number * (N - 1) * 4)
    elif gate == "chainXY" or gate == "fcXY":
        thetal = 2 * np.pi * np.random.rand(layer_number * (N + 1))
    # theta = np.zeros((N-1, layer_number))
    #phi = np.random.rand(N)
    #save the initial value for experimental

    ##Directly, suitble for exhaustive method###
    samples = gen_btsl(N)

    if join:
        a = np.zeros(len(samples))
        phi = np.exp(a) / np.sum(np.exp(a))
        para = np.concatenate((a, thetal))
    else:
        a = np.zeros(N)
        phi = 1 / (1 + np.exp(-a))
        para = np.concatenate((a, thetal))


    lossfl = []
    def call_back(x):
        lossf = loss_func(x, samples, beta, H, layer_number,  gate=gate, samp=samp, join=join, noise=noise)
        lossfl.append(lossf)
        pid = os.getpid()
        print("process: ", pid, "Current loss: ", lossf)

    loss_func_p = partial(loss_func,  gate=gate, samp=samp, join=join, noise=noise)
    if method == "lbfgs":
        results = scipy.optimize.minimize(loss_func_p, para, jac=grad,   method="l-bfgs-b", args=(samples, beta, H, layer_number), tol = 1e-15, options={"maxiter" : 700, "disp" : True}, callback=call_back)
    elif method == "nelder-mead" or (method == "Powell") or (method == "COBYLA"):
        results = scipy.optimize.minimize(loss_func_p, para, method=method, args=(samples, beta, H, layer_number), tol = 1e-15, options={"maxiter" : 1000, "disp" : True, "adaptive" : True}, callback=call_back)
    # elif method == "basinhop":
    #     results = scipy.optimize.basinhopping(loss_func_p, para, args=(samples, beta, H, layer_number), niter=niter)

        
    para = results.x
    #phi = para[:N]
    if join:
        Nphi = len(samples)
        phi = np.exp(para[:Nphi])/np.sum(np.exp(para[:Nphi]))
        thetal = para[Nphi:]
    else:
        phi = 1 / (1 + np.exp(-para[:N]))
        thetal = para[N:]

    return para, phi, thetal, lossfl
    

    # #One step, suitble for sample method
    # for i in range(niter):
    #     print("Currunt interation: %d/%d" % (i+1, niter))
    #     if samp:
    #         samples = gen_samples(phi, nbatch[i], N)

    #     lossf = loss_func(para, samples, beta, H, layer_number, gate, samp)
    #     print("Currunt loss: ", lossf)
    #     lossfl.append(lossf)
    #     results = scipy.optimize.minimize(loss_func, para, jac= grad, method="Nelder-Mead", args=(samples, beta, H, layer_number, gate, samp), tol = 1e-5, options={"maxiter" : 1, "disp" : True})
    #     para = results.x
    #     #phi = para[:N]
    #     phi = 1/(1+np.exp(-para[:N]))
    #     thetal = para[N:]

    # return para, phi, thetal, lossfl


def optimize_adam(niter, layer_number, beta, H, samples, nbatch = 100, gate="XY", lr=0.1, samp=False, join=False, decay=0, noise=False, peierls = False, symmetry = False, decoherence = False):
    np.random.seed()
    N = H.N
    
    if gate == "XY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N-1)*4)
    elif gate == "chainXY" or gate == "fcXY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N+1))
    
    if join:
        a = np.zeros(len(samples))
        phi = np.exp(a) / np.sum(np.exp(a))
        para = np.concatenate((a, thetal))
    else:
        a = np.zeros(N)
        phi = 1 / (1 + np.exp(-a))
        para = np.concatenate((a, thetal))

    print(len(samples))
    ###If use peierls loss
    if peierls:
        para = thetal
  
    b1 = 0.9
    b2 = 0.999
    e = 0.00000001
    mt = np.zeros(len(para))
    vt = np.zeros(len(para))

    lossfl = []

    for i in range(niter):
        print("Current interation: %d/%d" % (i+1, niter))
        if samp:
            samples = gen_samples(phi, nbatch, N)
            #samples = gen_btsl_sub(N, int(2**N))
        
        # totalbasis = gen_btsl(N)
        # samples = gen_btsl_sub(N, 16)

        if peierls:
            lossf = loss_func_peierls(para, samples, beta, H, layer_number,  nbatch=nbatch,  noise=noise,  gate = gate, samp = samp, join = join,  symmetry = symmetry, decoherence = decoherence)
        else:
            lossf = loss_func(para, samples, beta, H, layer_number,  nbatch = nbatch, gate = gate, samp = samp, join = join, noise = noise, symmetry = symmetry, decoherence = decoherence)

        #lossft = loss_func(para, totalbasis, beta, H, layer_number,  nbatch = nbatch, gate = gate, samp = samp, join = join, noise = noise, symmetry = symmetry)
        pid = os.getpid()
        print("process: ", pid, "Current loss: ", lossf)
        lossfl.append(lossf)
       
        grads = grad(para, samples, beta, H, layer_number, gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry, decoherence = decoherence)
        mt = b1*mt + (1-b1)*grads
        vt = b2*vt + (1-b2)*grads**2
        mtt = mt/(1-b1**(i+1))
        vtt = vt/(1-b2**(i+1))
        ###learning rate decay####
        if i > 20:
            print("decay")
            lr = decay

        para = para - lr*mtt/(np.sqrt(vtt)+e)

        if join:
            Nphi = len(samples)
            phi = np.exp(para[: Nphi])/np.sum(np.exp(para[:Nphi]))
            thetal = para[Nphi:]
        else:
            phi = 1/(1+np.exp(-para[:N]))
            thetal = para[N:]

    return para, phi, thetal, lossfl

def optimize_direct(niter, layer_number, beta, H, samples, nbatch = 100, gate="XY",  samp=False, join=False, noise=False, symmetry=False, peierls=False, decoherence=False):
    N = H.N 

    dim_theta = layer_number * (N + 1)
    # dim = N + dim_theta
    # bounds = [[0, 1]] * N + [[0, 2 * np.pi]] * (dim_theta)

    
    dim = dim_theta
    bounds = [[0, 2 * np.pi]] * (dim_theta)
     
    loss_func_peierls_p = partial(loss_func_peierls,  nbatch = nbatch,  gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry, decoherence = decoherence)

    res = scipydirect.minimize(loss_func_peierls_p, bounds=bounds, nvar = dim, args=(samples, beta, H, layer_number), algmethod=1, maxf = 80000, maxT=6000)
    print(res)
    
    return res.x

def optimize_zoopt(niter, layer_number, beta, H, samples, nbatch = 100, gate="XY",  samp=False, join=False, noise=False, symmetry = False, peierls = False, decoherence = False):
    np.random.seed()
    N = H.N
    lossfl = []

    def objfunction(solution):
        x = solution.get_x()
        if peierls:
            value = loss_func_peierls(np.array(x), samples, beta, H, layer_number, nbatch = nbatch,  gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry, decoherence = decoherence)
        else: 
            value = loss_func_bound(np.array(x), samples, beta, H, layer_number, nbatch = nbatch,  gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry,decoherence = decoherence)
           
        return value
    
    dim_theta = layer_number * (N + 1)
    # iniphi =  [0.5]*N
    # guess = [Solution(x = iniphi + (2*np.pi*np.random.rand(dim_theta)).tolist()) for i in range(5)]

    dim = N + dim_theta   # dimension
    obj = Objective(objfunction, Dimension(dim, [[0, 1]]*N + [[0, 2*np.pi]]*(dim_theta), [True]*dim))

    #peierls
    if peierls:
        dim = dim_theta 
        obj = Objective(objfunction, Dimension(dim, [[0, 2*np.pi]]*(dim_theta), [True]*dim))


    parameter = Parameter(budget=10000, uncertain_bits = 1,  exploration_rate = 0.02, parallel = True, server_num = 6)
    # parameter.set_train_size(22)
    # parameter.set_positive_size(2)
    sol = Opt.min(obj, parameter)
    #sol = Opt.min(obj, Parameter(budget=100*dim, parallel=True, server_num=4))
    # solution_list = ExpOpt.min(obj, parameter, repeat = 4, best_n = 4, plot=True, plot_file = "opt_progress.pdf")

    para = np.array(sol.get_x())
    if join:
        Nphi = len(samples)
        phi = para[:Nphi]
        thetal = para[Nphi:]      
    else:
        phi = para[:N]
        thetal = para[N:]
    lossfl = obj.get_history_bestsofar()
    
    return para, phi, thetal, lossfl

def optimize_gpso(niter, layer_number, beta, H, samples, nbatch = 100,  gate="XY",  samp=False, join=False, noise=False, symmetry = False):
    N = H.N
    samples = gen_btsl(N)
    loss_func_p = partial(loss_func_bound,  nbatch = nbatch, gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry)
    dim_theta = layer_number*(N + 1)
    dim = N + dim_theta
    bound = np.array([[0.001, 0.99]]*N + [[0, 2*np.pi]]*(dim_theta))
    para, lossfl = minimize(loss_func_p, args = (samples, beta, H, layer_number), dim = dim, bound = bound, boundary_handling = "periodic", popsize = 20, max_iter = 1000, wmin = 0.1, wmax = 0.5, c1 = 1.0, c2 = 0.5, c3 = 0.0, vp = 0.1, decay = False)

    if join:
        Nphi = len(samples)
        phi = para[:Nphi]
        thetal = para[Nphi:]      
    else:
        phi = para[:N]
        thetal = para[N:]
    print(para)
    return para, phi, thetal ,lossfl

