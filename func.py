##func

import numpy as np
from error_gate import *
from scipy.linalg import expm
import scipy.optimize
from functools import partial
import os
from zoopt import Dimension, Objective, Parameter, Opt
###General gate#####

si = np.array([[1.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j]])

sx = np.array([[0.+0.j, 1.+0.j],
               [1.+0.j, 0.+0.j]])

sy = np.array([[0., -1.j],
               [1.j, 0.]]) 

sz = np.array([[1.+0.j, 0.+0.j],    
               [0.+0.j, -1.+0.j]])

Len_theta = 0.05
Len_phi = 0.05

ERROR_XY_LIST = []

for i in range(6):
    Dtheta = (np.random.random()-0.5)/0.5*Len_theta
    Dphim = np.sqrt(Len_phi**2 - (Len_phi*Dtheta)**2/Len_theta**2)
    Dphi = (np.random.random()-0.5)/0.5*Dphim
    ERROR_XY_LIST.append((Dtheta, Dphi))

#print("ERROR_XY_LIST: ", ERROR_XY_LIST)

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


### For state vector#####
spin = [np.array([1., 0.]), np.array([0., 1.])]

def gen_btsl(N):

    basisN = int(2**N)
    btsl = [get_basis(ind, N) for ind in range(basisN)]
    return btsl

def jointprob(btsl, phi):
    prob = []
    for x in btsl:
        singprob = phi**(x)*(1-phi)**(1-x)
        prob.append(reduce(lambda a, b: a*b, singprob, 1))
    
    return np.array(prob)


####Extend N-qubit gate###


def Id(N):
    return tensorl([si for i in range(N)])



######variational gate########
def sq_gate(para, noise):
    return np.dot(rz(para[2], noise), np.dot(ry(para[1], noise), rz(para[0], noise)))

def SU4_gate(para, noise):
    SU4 = np.kron(sq_gate(para[:3], noise), si)
    SU4 = np.dot(np.kron(si, sq_gate(para[3:6], noise)),SU4)
    SU4 = np.dot(cnot(1), SU4)
    SU4 = np.dot(np.kron(rz(para[7], noise), si), SU4)
    SU4 = np.dot(np.kron(si, ry(para[8], noise)), SU4)
    SU4 = np.dot(cnot(0), SU4)
    SU4 = np.dot(np.kron(si, ry(para[9], noise)), SU4)
    SU4 = np.dot(cnot(1), SU4)
    SU4 = np.dot(np.kron(sq_gate(para[9:12], noise), si),SU4)
    SU4 = np.dot(np.kron(si, sq_gate(para[12:], noise)), SU4)

    return SU4

def singlegates(para, N, singlegate, noise, dthedphi = None):
    gatel = []
    if singlegate == "X":
        gatel = [rx(para[i], noise) for i in range(N)]
    elif singlegate == "Y":
        gatel = [ry(para[i], noise) for i in range(N)]
    elif singlegate == "Z":
        gatel = [rz(para[i], noise) for i in range(N)]
    return tensorl(gatel)    

def twoXXgate(i,N):
    oplist = [si for n in range(N)]
    oplist[i], oplist[i+1] = sx, sx
    return tensorl(oplist)

def twoYYgate(i,N):
    oplist = [si for n in range(N)]
    oplist[i], oplist[i+1] = sy, sy
    return tensorl(oplist)

def twoZZgate(i,N):
    oplist = [si for n in range(N)]
    oplist[i], oplist[i+1] = sz, sz
    return tensorl(oplist)
            
def enhan_XYgate(para, N, singlegate, noise, dthedphi=None):
    d = 2**N
    xylist = [twoXXgate(i,N)+twoYYgate(i,N) for i in range(N-1)]
    #xylist = [np.zeros((d,d)) for i in range(N-1)]
    xy = np.sum(xylist,axis=0)
    globalgate = expm(-1j*para[N]*xy)
    return np.dot(globalgate, singlegates(para, N, singlegate, noise, dthedphi))
    
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

def all_XYgate(para, N, singlegate, noise, dthedphi=None):
    d = 2**N
    xy = np.zeros((d,d), dtype = np.complex128)
    for i in range(N-1):
        for j in range(i+1,N):
            xy += arb_twoXXgate(i,j,N)+arb_twoYYgate(i,j,N)
    globalgate = expm(-1j*para[N]*xy)
    return np.dot(globalgate, singlegates(para, N, singlegate, noise, dthedphi))
    
    
    

def XYgate_grad(para, a, b, noise, dthedphi=None):
    XY_grad = np.kron(rx(para[0], noise), ry(para[1], noise))
    xy = rxy(para[2]+a, noise, dthedphi=dthedphi) 
    XY_grad = np.dot(xy, XY_grad)
    XY_grad = np.dot(np.kron(sx, si), XY_grad)
    XY_grad = np.dot(rxy(b, noise, dthedphi=dthedphi), XY_grad)
    XY_grad = np.dot(np.kron(sx, si), XY_grad)
    XY_grad = np.dot(np.kron(rz(para[3], noise), si), XY_grad)

    return XY_grad


def get_U_grad(thetal, layer_number, N, gate, gi, a, b, noise):
    theta = thetal.reshape((layer_number, N-1, 4))
    layer_ind = gi//((N-1)*4)
    bit_ind = gi//4%(N-1)
    I = Id(N)

    U = I
    for l in range(theta.shape[0]):
        Ul = I
        for i in range(theta.shape[1]):
            oplist = [si for n in range(N-1)]
            if (l == layer_ind) and (i == bit_ind):
                oplist[i] = XYgate_grad(theta[l, i, :], a, b, noise)
            else:
                oplist[i] = enhan_XYgate(theta[l, i, :], noise)

            Ul = np.dot(tensorl(oplist), Ul)
        U = np.dot(Ul, U)

    return U
    


def get_U(thetal, layer_number, N, singlegate, gate, noise):
    if gate == "XY":
        theta = thetal.reshape((layer_number, N+1))
        tq_gate = enhan_XYgate
    elif gate == "SU4":
        theta = thetal.reshape((layer_number, N-1, 15))
        tq_gate = SU4_gate
    elif gate == "all_XY":
        theta = thetal.reshape((layer_number, N+1))
        tq_gate = all_XYgate
    I = Id(N)

    U = I
    for l in range(theta.shape[0]): # layer
        Ul = I
        Ul = tq_gate(theta[l,:], N, singlegate, noise, dthedphi=None)
        U = np.dot(Ul, U)
    return U


def hamil_Ising(h, N):
    sx_list, sy_list, sz_list = Nbit_single(N)
    H = 0.+0.j
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

def q_expect(U, x,  beta, H):
    psi = tensorl([spin[int(i)] for i in x])
    psi = np.dot(U, psi)
    return beta*np.real(np.dot(np.conj(psi).T, np.dot(H, psi)))

def q_expect_exp(H, N, U, x, beta, h, basez, basezz, noise, Htype):
    ###prepare measue
    if Htype == "XY":
        psi = tensorl([spin[int(i)] for i in x])
        N = int(np.log2(len(psi)))
        ##Using total operator####
        psi = np.dot(U, psi)
        prob = np.abs(psi)**2
        ez = np.dot(prob, basez)
    
        psix = np.dot(tensorl([ry(np.pi/2, False) for i in range(N)]), psi)
        prob = np.abs(psix)**2
        exx = np.dot(prob, basezz)
    
        psiy = np.dot(tensorl([rx(np.pi/2, False) for i in range(N)]), psi)
        prob = np.abs(psiy)**2
        eyy = np.dot(prob, basezz)
    
    
        # ###Using local sum######
    
        # ### measure sum_sz
        # psi = np.dot(U, psi)
        # prob = np.abs(psi)**2
        # ez = measure_singlebit(N, prob, basez)
    
        # ### measure sum_sx*sx 
        # psix = np.dot(tensorl([ry(np.pi/2, False) for i in range(N)]), psi)
        # prob = np.abs(psix)**2
        # exx = measure_nncorr(N, prob, basezz)
    
        # ##measure sum_sy*sy
    
        # psiy = np.dot(tensorl([rx(np.pi/2, False) for i in range(N)]), psi)
        # prob = np.abs(psiy)**2
        # eyy = measure_nncorr(N, prob, basezz)
    
        H = -h*ez-exx-eyy
    
        return beta*H 
    if Htype == "Ising":
        psi = tensorl([spin[int(i)] for i in x])
        N = int(np.log2(len(psi)))
        ##Using total operator####
        psi = np.dot(U, psi)
        psix = np.dot(tensorl([ry(np.pi/2, False) for i in range(N)]), psi)
        prob = np.abs(psix)**2
        ex = np.dot(prob, basez)
    
        prob = np.abs(psi)**2
        ezz = np.dot(prob, basezz)
    
        H = -h*ex-ezz
        return beta*H
        


def loss_quantum(thetal, alpha, samples, beta, H, N, singlegate, layer_number, Htype, gate, noise=False, samp=False, join=False):
    h = 0.5
    basez = get_baseglobalz(N)
    basezz = get_baseglobalzz(N)

    if join:
        phi = np.exp(alpha)/np.sum(np.exp(alpha))
    else:
        phi = 1/(1+np.exp(-alpha))
    
    U = get_U(thetal, layer_number, N, singlegate, gate, noise)
    q_expectl = [q_expect_exp(H, N, U, x, beta, h, basez, basezz, noise, Htype) for x in samples]

    #q_expectl = [q_expect(U, x, beta, H) for x in samples]

    if samp: # Using samp
        return np.mean(q_expectl)
    else:
        if join:
            return np.dot(phi, q_expectl)
        else:
            prob = jointprob(samples, phi)
            return np.dot(prob, q_expectl)


def loss_quantum_grad(thetal, alpha, samples, beta, H, N, layer_number,  gi, a, b, noise=False, gate="XY", samp=False, join=False):

    h = 0.5
    basez = get_baseglobalz(N)
    basezz = get_baseglobalzz(N)

    if join:
        phi = np.exp(alpha)/np.sum(np.exp(alpha))
    else:
        phi = 1/(1+np.exp(-alpha))
    
    U = get_U_grad(thetal, layer_number, N, gate, gi, a, b, noise)
    #q_expectl = [q_expect(U, x, beta, H) for x in samples]
    q_expectl = [q_expect_exp(H, N, U, x, beta, h, basez, basezz, noise, Htype) for x in samples]

    if samp: # Using samp
        return np.mean(q_expectl)
    else:
        if join:
            return np.dot(phi, q_expectl)
        else:
            prob = jointprob(samples, phi)
            return np.dot(prob, q_expectl)
    


def loss_func(para, samples, beta, H, N, singlegate, layer_number, Htype, gate, noise=False, samp=False, join=False):
    h = 0.5
    basez = get_baseglobalz(N)
    basezz = get_baseglobalzz(N)

    
    #phi = para[:N]
    if join:
        phi = np.exp(para[:int(2**N)])/np.sum(np.exp(para[:int(2**N)]))
        thetal = para[int(2**N):]
        U = get_U(thetal, layer_number, N, singlegate, gate, noise)
        loss_samp = np.log(phi)+[q_expect(U, x, beta, H) for x in samples]
        return np.dot(phi, loss_samp)

    else:
        phi = 1/(1+np.exp(-para[:N]))
        thetal = para[N:]
        U = get_U(thetal, layer_number, N, singlegate, gate, noise)
        #loss_samp = [logp(phi, x) + q_expect(U, x, beta, H) for x in samples]
        loss_samp = [logp(phi, x) + q_expect_exp(H, N, U, x, beta, h, basez, basezz, noise, Htype) for x in samples]
        if samp:
            return np.mean(loss_samp)
        else:
            prob = jointprob(samples, phi)
            return np.dot(prob, loss_samp)
        
def loss_func_zoopt(para, samples, beta, H, N, singlegate, layer_number, Htype, gate, noise=False, samp=False, join=False):
    h = 0.5
    basez = get_baseglobalz(N)
    basezz = get_baseglobalzz(N)

    if join:
        phi = np.exp(para[:int(2**N)])/np.sum(np.exp(para[:int(2**N)]))
        thetal = para[int(2**N):]
        U = get_U(thetal, layer_number, N, singlegate, gate, noise)
        loss_samp = np.log(phi)+[q_expect(U, x, beta, H) for x in samples]
        return np.dot(phi, loss_samp)

    else:
        #phi = para[:N]
        phi = 1/(1+np.exp(-para[:N]))
        thetal = para[N:]
        U = get_U(thetal, layer_number, N, singlegate, gate, noise)
        #loss_samp = [logp(phi, x) + q_expect(U, x, beta, H) for x in samples]
        loss_samp = [logp(phi, x) + q_expect_exp(H, N, U, x, beta, h, basez, basezz, noise, Htype) for x in samples]
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





def grad(para, samples, beta, H, N, singlegate, layer_number, Htype, gate, samp=False, join=False, noise=False):
    #phi = para[:N]

    h = 0.5
    basez = get_baseglobalz(N)
    basezz = get_baseglobalzz(N)

    if join:
        ##total join prob
        alpha = para[:int(2**N)]
        phi = np.exp(alpha)/np.sum(np.exp(alpha))
        thetal = para[int(2**N):]
        U = get_U(thetal, layer_number, N, singlegate, gate, noise)
        grad_logp = -np.outer(phi, phi)
        for i in range(len(phi)):
            grad_logp[i, i] = phi[i]-phi[i]**2
        
        #fx = np.log(phi) + np.array([q_expect(U, x, beta, H) for x in samples])
        fx = np.log(phi) + np.array([q_expect_exp(H, N, U, x, beta, h, basez, basezz, noise, Htype) for x in samples])
        grad_phi = np.dot((1+fx), grad_logp)


    else:
        alpha = para[:N]
        phi = 1/(1+np.exp(-alpha))
        thetal = para[N:]
        prob = jointprob(samples, phi)
        if not samp:
            prob = jointprob(samples,phi)
        else:
            b = loss_func(para, samples, beta, H, N, singlegate, layer_number, Htype, gate, samp, join, noise)
        
        U = get_U(thetal, layer_number, N, singlegate, gate, noise)
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
                fx = logp(phi, x) + q_expect_exp(H, N, U, x, beta, h, basez, basezz, noise, Htype)
                grad_phil.append(fx*(grad_logp))
            grad_phi = np.dot(prob, grad_phil)

    loss_quantum_p = partial(loss_quantum, samp=samp, join=join, noise = noise)
    grad_theta = scipy.optimize.approx_fprime(thetal, loss_quantum_p, 1e-8, alpha,  samples, beta, H, N, singlegate, layer_number, Htype, gate) 
    
    #grad_theta = grad_parashift(thetal, alpha, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=noise)

    #print("difference: ", grad_theta1-grad_theta)
                        #total 93
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
    np.random.seed()
    #bounds = [(0, 1)]*N+[(None, None)]*(N-1)*layer_number*15
    if gate == "XY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N-1)*4)
    elif gate == "SU4":
        thetal = 2*np.pi*np.random.rand(layer_number*(N-1)*15)
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
    elif method == "Nelder-Mead" or (method == "Powell") or (method == "COBYLA"):
        results = scipy.optimize.minimize(loss_func_p, para, method=method, args=(samples, beta, H, N, layer_number), tol = 1e-15, options={"maxiter" : 700, "disp" : True}, callback=call_back)

        
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


def optimize_adam(niter, layer_number, beta, H, N, singlegate, h, nbatch, Htype, gate, lr=0.1, samp=False, join=False, decay=0, noise=False):
    np.random.seed()
    if gate == "XY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N+1))
    elif gate == "SU4":
        thetal = 2*np.pi*np.random.rand(layer_number*(N-1)*15)
    # theta = np.zeros((N-1, layer_number))
    #phi = np.random.rand(N)
    elif gate == "all_XY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N+1))
    
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
        print("Currunt interation: %d/%d" % (i+1, niter))
        if samp:
            samples = gen_samples(phi, nbatch[i], N)
        
        lossf = loss_func(para, samples, beta, H, N, singlegate, layer_number, Htype, gate, samp=samp, join=join, noise=noise)
        pid = os.getpid()
        print("process: ", pid, "Currunt loss: ", lossf)

        lossfl.append(lossf)
       
        grads = grad(para, samples, beta, H, N, singlegate, layer_number, Htype, gate, samp=samp, join=join, noise=noise)
        mt = b1*mt + (1-b1)*grads
        vt = b2*vt + (1-b2)*grads**2
        mtt = mt/(1-b1**(i+1))
        vtt = vt/(1-b2**(i+1))
        ###learning rate decay####
        if i > 50:
            print("decay")
            lr = decay

        para = para - lr*mtt/(np.sqrt(vtt)+e) #renew of para
        #print(para)
        #phi = para[:N]
        if join:
            phi = np.exp(para[:int(2**N)])/np.sum(np.exp(para[:int(2**N)]))
            thetal = para[int(2**N):]
        else:
            phi = 1/(1+np.exp(-para[:N]))
            thetal = para[N:]

    return para, phi, thetal, lossfl

def optimize_zoopt(niter, layer_number, beta, H, N, singlegate, h, nbatch, Htype, gate, lr=0.1, samp=False, join=False, decay=0, noise=False):
    np.random.seed()
    if gate == "XY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N+1))
    elif gate == "SU4":
        thetal = 2*np.pi*np.random.rand(layer_number*(N-1)*15)
    # theta = np.zeros((N-1, layer_number))
    #phi = np.random.rand(N)
    elif gate == "all_XY":
        thetal = 2*np.pi*np.random.rand(layer_number*(N+1))
    
    if join:
        a = np.zeros(int(2**N))
        phi = np.exp(a)/np.sum(np.exp(a))
        para = np.concatenate((a, thetal))
    else:
        a = np.zeros(N)
        phi = 1/(1+np.exp(-a))
        para = np.concatenate((a, thetal))


    lossfl = []
    samples = gen_btsl(N)
    def objfunction(solution):
        x = solution.get_x()
        value = loss_func_zoopt(np.array(x), samples, beta, H, N, singlegate, layer_number, Htype, gate, samp=samp, join=join, noise=noise)
        return value
    dim = N + layer_number*(N+1)   # dimension
    obj = Objective(objfunction, Dimension(dim, np.concatenate(([[-100, 100]]*N,[[0,100]]*(layer_number*(N+1))),axis = 0), [True]*dim))
    sol = Opt.min(obj, Parameter(budget=100*dim))
    #sol = Opt.min(obj, Parameter(budget=100*dim, parallel=True, server_num=4))
    
    para = np.array(sol.get_x())
    phi = para[:N]
    thetal = para[N:]
    lossfl = obj.get_history_bestsofar()
    
    return para, phi, thetal, lossfl









