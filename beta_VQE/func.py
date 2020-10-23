##func

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


###two bit XY gate error parameter###
Len_theta = 0.05
Len_phi = 0.05

ERROR_XY_LIST = []
for i in range(6):
    Dtheta = (np.random.rand()-0.5)/0.5*Len_theta
    Dphim = np.sqrt(Len_phi**2 - (Len_phi*Dtheta)**2/Len_theta**2)
    Dphi = (np.random.rand()-0.5)/0.5*Dphim
    ERROR_XY_LIST.append((Dtheta, Dphi))

print("ERROR_XY_LIST: ", ERROR_XY_LIST)


#Global  variable
N = config.N
Id = tensorl([si for i in range(N)])
Totbasis = gen_btsl(N)

Xylist = [arb_twoXXgate(i, i+1, N)+arb_twoYYgate(i, i+1, N) for i in range(N-1)]
XY = np.sum(Xylist,axis=0)

FCXY = np.zeros((2**N, 2**N), dtype = np.complex128)
for i in range(N-1):
    for j in range(i+1,N):
        FCXY += arb_twoXXgate(i,j,N)+arb_twoYYgate(i,j,N)

BasisBlock, BasisBlockInd = get_sym_basis(Totbasis)

Basez = get_baseglobalz(N)
Basezz = get_baseglobalzz(N)


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
        singprob = phi**(x)*(1-phi)**(1-x)
        prob.append(reduce(lambda a, b: a*b, singprob, 1))
    
    return np.array(prob)


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
        gatel = [rz(para[i], noise) for i in range(N)]
    
    singlelayer = tensorl(gatel)
    if not (qN == None):
        singlelayer = get_block(singlelayer, qN, BasisBlockInd)
    return singlelayer  

def chain_XY(theta, N, noise, qN = None):
    xy = XY
    if not (qN == None):
        xy = get_block(XY, qN, BasisBlockInd)
    
    return expm(-1j*theta*xy)

def fc_XY(theta, N, noise, qN = None):
    fcxy = FCXY
    if not (qN == None):
        fcxy = get_block(FCXY, qN, BasisBlockInd)

    return  expm(-1j*theta*fcxy)


def XYgate_grad(para, a, b, noise, dthedphi=None):
    XY_grad = np.kron(rz(para[0], noise), rz(para[1], noise))
    xy = rxy(para[2]+a, noise, dthedphi=dthedphi) 
    XY_grad = np.dot(xy, XY_grad)
    XY_grad = np.dot(np.kron(sx, si), XY_grad)
    XY_grad = np.dot(rxy(b, noise, dthedphi=dthedphi), XY_grad)
    XY_grad = np.dot(np.kron(sx, si), XY_grad)
    XY_grad = np.dot(np.kron(rz(para[3], noise), si), XY_grad)

    return XY_grad


def get_U(thetal, layer_number, N, gate, noise, qN = None):
    if noise and (not qN == None):
        print("Warning: Currently symmetry block must be used without noise!")
        
    if gate == "XY":
        theta = thetal.reshape((layer_number, N-1, 4))
        U = 1.0
        for l in range(theta.shape[0]): # layer
            Ul = 1.0
            for i in range(theta.shape[1]):
                oplist = [si for n in range(N-1)] 
                oplist[i] = enhan_XYgate(theta[l, i, :], noise, dthedphi=ERROR_XY_LIST[i])
                tq_gateN = tensorl(oplist)
                Ul = np.dot(tq_gateN , Ul)
            U = np.dot(Ul, U)
        
    elif gate == "chainXY":
        theta = thetal.reshape((layer_number, N+1))
        U = 1.0
        for l in range(theta.shape[0]):
            Ul = singlelayer(theta[l, :N], N, "Z", noise, qN = qN)
            Ul = np.dot(chain_XY(theta[l, N], N, noise, qN = qN), Ul)
            U = np.dot(Ul, U)

    elif gate == "fcXY":
        theta = thetal.reshape((layer_number, N+1))
        U = 1.0
        for l in range(theta.shape[0]):
            Ul = singlelayer(theta[l, :N], N, "Z", noise, qN = qN)
            Ul = np.dot(fc_XY(theta[l, N], N, noise, qN = qN), Ul)
            U = np.dot(Ul, U)

    return U

def get_U_grad(thetal, layer_number, N, gate, gi, a, b, noise):
    theta = thetal.reshape((layer_number, N-1, 4))
    layer_ind = gi//((N-1)*4)
    bit_ind = gi//4%(N-1)

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



def hamil_Ising(h, N):
    sx_list, sy_list, sz_list = Nbit_single(N)
    H = 0
    for i in range(N):
        H += h*sx_list[i]

    for i in range(N-1):
        H += np.dot(sz_list[i], sz_list[i+1])
        
    return -H

def hamil_XY(h, N):
    sx_list, sy_list, sz_list = Nbit_single(N) 
    H = 0.+0.j
    for i in range(N):
        H += h*sz_list[i]
    
    for i in range(N-1):
        H += np.dot(sx_list[i], sx_list[i+1])+np.dot(sy_list[i], sy_list[i+1])
    
    return -H



def gen_samples(phi, nbatch, N):
    samples = np.zeros((nbatch, N))
    for i in range(nbatch):
        np.random.seed()
        samples[i, :] = phi > np.random.rand(N)
    
    return samples


def logp(phi, x):
    return np.sum(x*np.log(phi)+(1-x)*np.log(1-phi))

def q_expect(Ul, x,  beta, H):
    #Ul is list contain all qN block
    qN = np.sum(x)
    U = Ul[qN]
    psi = tensorl([spin[int(i)] for i in x])
    psi = get_block(psi, qN, BasisBlockInd)
    H = get_block(H, qN, BasisBlockInd)
    psi = np.dot(U, psi)

    return beta*np.real(np.dot(np.conj(psi).T, np.dot(H, psi))/np.dot(np.conj(psi).T, psi))


def q_expect_exp(U, x, beta, h,  noise):
    ###prepare measue
    psi = tensorl([spin[int(i)] for i in x])
    N = int(np.log2(len(psi)))
    ##Using total operator####
    psi = np.dot(U, psi)
    prob = np.abs(psi)**2
    prob = generateRandomProb_s(prob, stats = 10000)
    ez = np.dot(prob, Basez)/np.sum(prob)

    psix = np.dot(tensorl([ry(np.pi/2, noise) for i in range(N)]), psi)
    prob = np.abs(psix)**2
    prob = generateRandomProb_s(prob, stats = 10000)
    
    exx = np.dot(prob, Basezz)/np.sum(prob)

    psiy = np.dot(tensorl([rx(np.pi/2, noise) for i in range(N)]), psi)
    prob = np.abs(psiy)**2
    prob = generateRandomProb_s(prob, stats = 10000)

    eyy = np.dot(prob, Basezz)/np.sum(prob)

    H = -h*ez-exx-eyy
    return beta*H

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


def loss_quantum(thetal, alpha, samples, beta, H, N, layer_number, h = 0.5,  noise=False,  gate="XY", samp=False, join=False, symmetry = True):
    
    if join:
        phi = np.exp(alpha)/np.sum(np.exp(alpha))
    else:
        phi = 1/(1+np.exp(-alpha))
    

    if symmetry:
        Ul = []
        for n in range(N+1):
            Ul.append(get_U(thetal, layer_number, N, gate, noise, qN = n))
        q_expectl = [q_expect(Ul, x, beta, H) for x in samples]
    else:
        U = get_U(thetal, layer_number, N, gate, noise)
        q_expectl = [q_expect_exp(U, x, beta, h, noise) for x in samples]

    if samp: # Using samp
        return np.mean(q_expectl)
    else:
        if join:
            return np.dot(phi, q_expectl)
        else:
            prob = jointprob(samples, phi)
            return np.dot(prob, q_expectl)


def loss_quantum_grad(thetal, alpha, samples, beta, H, N, layer_number,  gi, a, b, h = 0.5, noise=False, gate="XY", samp=False, join=False):

    if join:
        phi = np.exp(alpha)/np.sum(np.exp(alpha))
    else:
        phi = 1/(1+np.exp(-alpha))
    
    U = get_U_grad(thetal, layer_number, N, gate, gi, a, b, noise)
    #q_expectl = [q_expect(U, x, beta, H) for x in samples]
    q_expectl = [q_expect_exp(U, x, beta, h, noise) for x in samples]

    if samp: # Using samp
        return np.mean(q_expectl)
    else:
        if join:
            return np.dot(phi, q_expectl)
        else:
            prob = jointprob(samples, phi)
            return np.dot(prob, q_expectl)
    


def loss_func(para, samples, beta, H, N, layer_number, h = 0.5,  noise=False, gate="XY", samp=False, join=False, symmetry = True):
    
    #phi = para[:N]
    if join:
        phi = np.exp(para[:int(2**N)])/np.sum(np.exp(para[:int(2**N)]))
        thetal = para[int(2**N):]
        U = get_U(thetal, layer_number, N, gate, noise)
        loss_samp = np.log(phi)+[q_expect(U, x, beta, H) for x in samples]
        return np.dot(phi, loss_samp)

    else:
        phi = 1/(1+np.exp(-para[:N]))
        thetal = para[N:]

        if symmetry:
            Ul = []
            for n in range(N+1):
                Ul.append(get_U(thetal, layer_number, N, gate, noise, qN = n))
            loss_samp = [logp(phi, x) + q_expect(Ul, x, beta, H) for x in samples]

        else:
            U = get_U(thetal, layer_number, N, gate, noise)
            loss_samp = [logp(phi, x) + q_expect_exp(U, x, beta, h, noise) for x in samples]

        if samp:
            return np.mean(loss_samp)
        else:
            prob = jointprob(samples, phi)
            return np.dot(prob, loss_samp) 
 
def loss_func_bound(para, samples, beta, H, N, layer_number, h = 0.5,  gate="XY",  noise=False, samp=False, join=False, symmetry = True):
    if join:
        phi = np.exp(para[:int(2**N)])/np.sum(np.exp(para[:int(2**N)]))
        thetal = para[int(2**N):]
        U = get_U(thetal, layer_number, N, gate, noise)
        loss_samp = np.log(phi)+[q_expect(U, x, beta, H) for x in samples]
        return np.dot(phi, loss_samp)

    else:
        phi = para[:N]
        thetal = para[N:]
        if symmetry:
            Ul = []
            for n in range(N+1):
                Ul.append(get_U(thetal, layer_number, N, gate, noise, qN = n))
            loss_samp = [logp(phi, x) + q_expect(Ul, x, beta, H) for x in samples]

        else:
            U = get_U(thetal, layer_number, N, gate, noise)
            loss_samp = [logp(phi, x) + q_expect_exp(U, x, beta, h, noise) for x in samples]

        if samp:
            return np.mean(loss_samp)
        else:
            prob = jointprob(samples, phi)
            return np.dot(prob, loss_samp) 

##parametershift for grad#######

def grad_parashift(thetal, alpha,  samples, beta,  H, N, layer_number, gate="XY", samp=False, join=False, noise=False):
    grad_theta = []
    dth = np.zeros(len(thetal))
    for i in range(len(thetal)):
        if not (i % 4 == 2):
            ###parameter shift for single-qubit gate
            dth[i] = np.pi/2
            loss1 = loss_quantum(thetal+dth, alpha, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=noise)
            dth[i] = -np.pi/2
            loss1 -= loss_quantum(thetal+dth, alpha, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=noise)
            grad_theta.append(loss1*0.5)
            dth[i] = 0

        else:
            ####parameter shift for XY gate
            lossX = loss_quantum_grad(thetal, alpha, samples, beta, H, N, layer_number,  i, np.pi/8, np.pi/8, gate=gate, samp=samp, join=join, noise=noise) - loss_quantum_grad(thetal, alpha, samples, beta, H, N, layer_number, i, -np.pi/8, -np.pi/8, gate=gate, samp=samp, join=join, noise=noise)
            
            lossY = loss_quantum_grad(thetal, alpha, samples, beta, H, N, layer_number, i, np.pi/8, -np.pi/8, gate=gate, samp=samp, join=join, noise=noise) - loss_quantum_grad(thetal, alpha, samples, beta, H, N, layer_number,  i, -np.pi/8, np.pi/8, gate=gate, samp=samp, join=join, noise=noise)

            grad_theta.append(lossX +lossY)
        
    return np.array(grad_theta)



def grad(para, samples, beta, H, N, layer_number, h=0.5,  gate="XY", samp=False, join=False, noise=False):
    #phi = para[:N]

    if join:
        ##total join prob
        alpha = para[:int(2**N)]
        phi = np.exp(alpha)/np.sum(np.exp(alpha))
        thetal = para[int(2**N):]
        U = get_U(thetal, layer_number, N, gate, noise)
        grad_logp = -np.outer(phi, phi)
        for i in range(len(phi)):
            grad_logp[i, i] = phi[i]-phi[i]**2
        
        #fx = np.log(phi) + np.array([q_expect(U, x, beta, H) for x in samples])
        fx = np.log(phi) + np.array([q_expect_exp(U, x, beta, h, noise) for x in samples])
        grad_phi = np.dot((1+fx), grad_logp)


    else:
        alpha = para[:N]
        phi = 1/(1+np.exp(-alpha))
        thetal = para[N:]
        prob = jointprob(samples, phi)
        if not samp:
            prob = jointprob(samples,phi)
        else:
            b = loss_func(para, samples, beta, H, N, layer_number, gate, samp, join, noise)
        
        U = get_U(thetal, layer_number, N, gate, noise)
        grad_phil = []
        if samp:
            #Using samples
            for x in samples:
                grad_logp = (x-phi)
                fx = logp(phi, x) + q_expect(U, x, beta, H)
                grad_phil.append((fx-b)*(grad_logp))
            grad_phi = np.mean(grad_phil)
        
        else:
            #exhaus
            for x in samples:
                grad_logp = (x-phi)
                #fx = logp(phi, x) + q_expect(U, x, beta, H)
                fx = logp(phi, x) + q_expect_exp(U, x, beta, h, noise)
                grad_phil.append(fx*(grad_logp))
            grad_phi = np.dot(prob, grad_phil)

    loss_quantum_p = partial(loss_quantum, gate = gate, samp=samp, join=join)
    grad_theta = scipy.optimize.approx_fprime(thetal, loss_quantum_p, 1e-8, alpha,  samples, beta,  H, N, layer_number) 
    
    # grad_theta = grad_parashift(thetal, alpha, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=noise)

    # print("difference: ", grad_theta1-grad_theta)
                        
    grad = np.concatenate((grad_phi, grad_theta))
    return grad



def exact(beta, H):
    e = np.linalg.eigvalsh(H)
    Z = np.sum(np.exp(-beta*e))
    F = -np.log(Z)/beta
    E = np.sum(np.dot(np.exp(-beta*e),e))/Z
    S = (E-F)*beta

    return F, E, S



def optimize_scip(niter, layer_number, beta, H, N, h, nbatch, gate="XY", samp=False, method='lbfgs', join=False, noise=False):
    #np.random.seed()
    #bounds = [(0, 1)]*N+[(None, None)]*(N-1)*layer_number*15
    if gate == "XY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N-1)*4)
    elif gate == "chainXY" or gate == "fcXY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N+1))
    # theta = np.zeros((N-1, layer_number))
    #phi = np.random.rand(N)
    if join:
        a = np.zeros(int(2**N))
        phi = np.exp(a)/np.sum(np.exp(a))
        para = np.concatenate((a, thetal))
    else:
        a = np.zeros(N)
        phi = 1/(1+np.exp(-a))
        para = np.concatenate((a, thetal))

    #save the initial value for experimental



    ##directly###
    samples = gen_btsl(N)

    lossfl = []
    def call_back(x):
        lossf = loss_func(x, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=noise)
        lossfl.append(lossf)
        pid = os.getpid()
        print("process: ", pid, "Currunt loss: ", lossf)

    loss_func_p = partial(loss_func, gate=gate, samp=samp, join=join, noise=noise)
    if method == "lbfgs":
        results = scipy.optimize.minimize(loss_func_p, para, jac=grad,   method="l-bfgs-b", args=(samples, beta, H, N, layer_number), tol = 1e-15, options={"maxiter" : 700, "disp" : True}, callback=call_back)
    elif method == "nelder-mead" or (method == "Powell") or (method == "COBYLA"):
        results = scipy.optimize.minimize(loss_func_p, para, method=method, args=(samples, beta, H, N, layer_number), tol = 1e-15, options={"maxiter" : 1000, "disp" : True, "adaptive" : True}, callback=call_back)
    # elif method == "basinhop":
    #     results = scipy.optimize.basinhopping(loss_func_p, para, args=(samples, beta, H, N, layer_number), niter=niter)

        
    para = results.x
    #phi = para[:N]
    if join:
        phi = np.exp(para[:int(2**N)])/np.sum(np.exp(para[:int(2**N)]))
        thetal = para[int(2**N):]
    else:
        phi = 1/(1+np.exp(-para[:N]))
        thetal = para[N:]

    return para, phi, thetal, lossfl
    

    # #one step
    # for i in range(niter):
    #     print("Currunt interation: %d/%d" % (i+1, niter))
    #     if samp:
    #         samples = gen_samples(phi, nbatch[i], N)

    #     lossf = loss_func(para, samples, beta, H, N, layer_number, gate, samp)
    #     print("Currunt loss: ", lossf)
    #     lossfl.append(lossf)
    #     results = scipy.optimize.minimize(loss_func, para, jac= grad, method="Nelder-Mead", args=(samples, beta, H, N, layer_number, gate, samp), tol = 1e-5, options={"maxiter" : 1, "disp" : True})
    #     para = results.x
    #     #phi = para[:N]
    #     phi = 1/(1+np.exp(-para[:N]))
    #     thetal = para[N:]

    # return para, phi, thetal, lossfl


def optimize_adam(niter, layer_number, beta, H, N,  h, nbatch, gate="XY", lr=0.1, samp=False, join=False, decay=0, noise=False):
    #np.random.seed()
    if gate == "XY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N-1)*4)
    elif gate == "chainXY" or gate == "fcXY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N+1))
        
    # theta = np.zeros((N-1, layer_number))
    #phi = np.random.rand(N)
    
    if join:
        a = np.zeros(int(2**N))
        phi = np.exp(a)/np.sum(np.exp(a))
        para = np.concatenate((a, thetal))
    else:
        a = np.zeros(N)
        phi = 1/(1+np.exp(-a))
        para = np.concatenate((a, thetal))
    # if os.path.exists("para.npy"):
    #     para = np.load("para.npy")
    #     phi = para[:N]
    #     theta = para[N:].reshape((layer_number, N-1))

    #alpha = 0.02
    b1 = 0.9
    b2 = 0.999
    e = 0.00000001
    mt = np.zeros(len(para))
    vt = np.zeros(len(para))

    lossfl = []
    samples = gen_btsl(N)

    for i in range(niter):
        print("Current interation: %d/%d" % (i+1, niter))
        if samp:
            samples = gen_samples(phi, nbatch[i], N)
        
        lossf = loss_func(para, samples, beta, H, N, layer_number, h = h, gate = gate, samp = samp, join = join, noise = noise)
        pid = os.getpid()
        print("process: ", pid, "Current loss: ", lossf)

        lossfl.append(lossf)
       
        grads = grad(para, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=noise)
        mt = b1*mt + (1-b1)*grads
        vt = b2*vt + (1-b2)*grads**2
        mtt = mt/(1-b1**(i+1))
        vtt = vt/(1-b2**(i+1))
        ###learning rate decay####
        if i > 20:
            print("decay")
            lr = decay

        para = para - lr*mtt/(np.sqrt(vtt)+e)
        #print(para)
        #phi = para[:N]
        if join:
            phi = np.exp(para[:int(2**N)])/np.sum(np.exp(para[:int(2**N)]))
            thetal = para[int(2**N):]
        else:
            phi = 1/(1+np.exp(-para[:N]))
            thetal = para[N:]

    return para, phi, thetal, lossfl

def optimize_zoopt(niter, layer_number, beta, H, N, h, nbatch, gate="XY",  samp=False, join=False, noise=False):
    #np.random.seed()

    lossfl = []
    samples = gen_btsl(N)
    def objfunction(solution):
        x = solution.get_x()
        value = loss_func_bound(np.array(x), samples, beta, H, N, layer_number,h = h, gate=gate, samp=samp, join=join, noise=noise)
        return value
    
    dim_theta = layer_number*(N+1)
    # iniphi =  [0.5]*N
    # guess = [Solution(x = iniphi + (2*np.pi*np.random.rand(dim_theta)).tolist()) for i in range(5)]

    dim = N + dim_theta   # dimension
    obj = Objective(objfunction, Dimension(dim, [[0, 1]]*N + [[np.pi/2, 2*np.pi]]*(dim_theta), [True]*dim))
    parameter = Parameter(budget=10000, uncertain_bits = 1,  exploration_rate=0.02, parallel=True, server_num=6)
    # parameter.set_train_size(22)
    # parameter.set_positive_size(2)
    sol = Opt.min(obj, parameter)
    #sol = Opt.min(obj, Parameter(budget=100*dim, parallel=True, server_num=4))
    # solution_list = ExpOpt.min(obj, parameter, repeat = 4, best_n = 4, plot=True, plot_file = "opt_progress.pdf")

    para = np.array(sol.get_x())
    phi = para[:N]
    thetal = para[N:]
    lossfl = obj.get_history_bestsofar()
    
    return para, phi, thetal, lossfl

def optimize_gpso(niter, layer_number, beta, H, N, h, nbatch, gate="XY",  samp=False, join=False, noise=False):
    samples = gen_btsl(N)
    loss_func_p = partial(loss_func_bound, h = h, gate=gate, samp=samp, join=join, noise=noise)
    dim_theta = layer_number*(N + 1)
    dim = N + dim_theta

    bound = np.array([[0.001, 0.99]]*N + [[0, 2*np.pi]]*(dim_theta))
    print(len(bound))
    para, lossfl = minimize(loss_func_p, args = (samples, beta, H, N, layer_number), dim = dim, bound = bound, boundary_handling = "periodic", popsize = 20, max_iter = 1000, wmin = 0.1, wmax = 0.5, c1 = 1.0, c2 = 0.5, c3 = 0.0, vp = 0.1, decay = False)

    phi = para[:N]
    thetal = para[N:]
    print(para)
    return para, phi, thetal ,lossfl
