import numpy as np
from func import *
import scipy.optimize
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
from functools import partial
#from scipydirect import minimize
import os
from zoopt import Dimension, Objective, Parameter, Opt, Solution, ExpOpt
from gpso import minimize
import config
from multiprocessing import Pool
import scipydirect 

# class Varcircuit():
#     def _init_(self, N, layer_number, gate="chanXY", noise=False, samp=False):
#         self.N = N 
#         self.layer_number = layer_number
#         self.noise = noise 
#         self.samp = samp 
#         self.gate = gate
    
#     def get_U(self, thetal, qN=None):
#         if self.gate == "XY":
#             theta = thetal.reshape((self.layer_number, self.N - 1, 4))
#             U = 1.0
#             for l in range(theta.shape[0]): # layer
#                 Ul = 1.0
#                 for i in range(theta.shape[1]):
#                     oplist = [si for n in range(N - 1)] 
#                     oplist[i] = enhan_XYgate(theta[l, i, :], self.noise, dthedphi=ERROR_XY_LIST[i])
#                     tq_gateN = tensorl(oplist)
#                     Ul = np.dot(tq_gateN , Ul)
#                 U = np.dot(Ul, U)
        
       
#         elif gate == "chainXY":
#             theta = thetal.reshape((self.layer_number, self.N))
#             U = 1.0
#             for l in range(theta.shape[0]):
#                 Ul = singlelayer(theta[l, :N], N, "Z", noise=self.noise, qN = qN)
#                 Ul = np.dot(chain_XY(np.pi / 8, N, noise=self.noise, qN=qN), Ul)
#                 U = np.dot(Ul, U)

#         elif gate == "fcXY":
#             theta = thetal.reshape((self.layer_number, self.N))
#             U = 1.0
#             for l in range(theta.shape[0]):
#                 Ul = singlelayer(theta[l, :N], N, "Z", self.noise, qN = qN)
#                 Ul = np.dot(fc_XY(np.pi / 8, N, self.noise, qN = qN), Ul)
#                 U = np.dot(Ul, U)

#         return U
        



class betaVQE():
    def __init__(self, N, beta, h, layer_number, niter=150, nbatch=10, Htype="XY", gate="chainXY", optimizer="adam", samp=False, join=False, noise=False, decay=0.05, lr=0.1, symmetry=False, peierls=False, decoherence=False, parallel=False):
        ''' Initial function'''
        self.N = N
        self.beta = beta
        self.h = h
        self.layer_number = layer_number
        self.niter = niter 
        self.nbatch = nbatch
        self.Htype = Htype
        self.gate = "chainXY" 
        self.optimizer = optimizer
        self.samp = samp 
        self.join = join 
        self.noise = noise 
        self.decay = decay
        self.lr = lr
        self.symmetry = symmetry 
        self.peierls = peierls 
        self.decoherence = decoherence 
        self.parallel = parallel
        self.H = Hamiltonian(h=self.h, N=self.N, Htype = self.Htype)
        self.var_basis = gen_btsl(N)
        self.sol = []
        self.traj = []
        self.xtraj = []
    
    def create_tag(self):
        tag = "Hamil-" + str(self.Htype)
        tag += "_N" + str(self.N)
        tag += "_beta" + str(self.beta)
        tag += "_h" + str(self.h)
        tag += "_layer_number" + str(self.layer_number)
        tag += "_method-" + str(self.optimizer)
        tag += "_gate-" + str(self.gate)
        tag += "_noise" if self.noise else ""
        tag += "_sym" if self.symmetry else ""
        tag += "_peierls" if self.peierls else ""
        tag += "_decoherence" if self.decoherence else ""
        if self.samp:
            tag += "_nbatch" + str(self.nbatch)
        
        if self.optimizer == "adam":
            tag += "_lr" + str(self.lr)
            tag += "_decay" + str(self.decay)
        
        return tag

    def get_U(self, thetal, qN=None):
        if self.gate == "XY":
            theta = thetal.reshape((self.layer_number, self.N - 1, 4))
            U = 1.0
            for l in range(theta.shape[0]): # layer
                Ul = 1.0
                for i in range(theta.shape[1]):
                    oplist = [si for n in range(N - 1)] 
                    oplist[i] = enhan_XYgate(theta[l, i, :], self.noise, dthedphi=ERROR_XY_LIST[i])
                    tq_gateN = tensorl(oplist)
                    Ul = np.dot(tq_gateN , Ul)
                U = np.dot(Ul, U)
        
       
        elif self.gate == "chainXY":
            theta = thetal.reshape((self.layer_number, self.N))
            U = 1.0
            for l in range(theta.shape[0]):
                Ul = singlelayer(theta[l, :N], N, "Z", noise=self.noise, qN = qN)
                Ul = np.dot(chain_XY(np.pi / 8, N, noise=self.noise, qN=qN), Ul)
                U = np.dot(Ul, U)

        elif self.gate == "fcXY":
            theta = thetal.reshape((self.layer_number, self.N))
            U = 1.0
            for l in range(theta.shape[0]):
                Ul = singlelayer(theta[l, :N], N, "Z", self.noise, qN = qN)
                Ul = np.dot(fc_XY(np.pi / 8, N, self.noise, qN = qN), Ul)
                U = np.dot(Ul, U)
        
        return U

    def get_U_grad(self, thetal, gi, a, b):
        '''Only for XY circuit '''
        theta = thetal.reshape((self.layer_number, self.N - 1, 4))
        layer_ind = gi // ((self.N - 1) * 4)
        bit_ind = gi // 4%(self.N - 1)

        U = Id
        for l in range(theta.shape[0]):
            Ul = Id
            for i in range(theta.shape[1]):
                oplist = [si for n in range(self.N-1)]
                if (l == layer_ind) and (i == bit_ind):
                    oplist[i] = XYgate_grad(theta[l, i, :], a, b, noise=self.noise, dthedphi=ERROR_XY_LIST[i])
                else:
                    oplist[i] = enhan_XYgate(theta[l, i, :], noise=self.noise,
                    dthedphi=ERROR_XY_LIST[i])

                Ul = np.dot(tensorl(oplist), Ul)
            U = np.dot(Ul, U)
        
        return U
    
    def evolve_rho(self, rho, thetal):
        theta = thetal.reshape((self.layer_number, self.N))
        for l in range(theta.shape[0]):
            sing_layer =  singlelayer(theta[l, :N], N, "Z", self.noise)
            sing_layer = qtp.Qobj(sing_layer, dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper')
            rho = sing_layer.dag() * rho * sing_layer
            #tlist = [0, -np.pi / 8 / g]
            tlist = np.linspace(0, -np.pi / 8 / g, 50)
            res = qtp.mesolve(Hxy, rho, tlist, c_ops = C_ops)
            rho = res.states[-1]

        return rho

    def q_expects(self, x, Ul):
        H = (self.H).H
        #Ul is list contain all qN block
        qN = int(np.sum(x))
        U = Ul[qN]
        psi = tensorl([spin[int(i)] for i in x])
        psi = get_block(psi, qN, BasisBlockInd)
        H = get_block(H, qN, BasisBlockInd)
        psi = np.dot(U, psi)

        return np.real(np.dot(np.conj(psi).T, np.dot(H, psi))/np.dot(np.conj(psi).T, psi))

    def q_expect_rho(self, x, thetal):
        N = self.N
        psi = qtp.tensor([qtp.basis(2, int(i)) for i in x])
        psi = qtp.Qobj(tensorl([spin[int(i)] for i in x]), dims = [[2]*N, [1]*N], shape = (2**N, 1), type='ket')
        psirho = qtp.ket2dm(psi)
        psirho = self.evolve_rho(psirho, thetal)
        
        H = qtp.Qobj((self.H).H,  dims = [[2]*N, [2]*N], shape = (2**N, 2**N), type='oper', isherm=True)
        Hexpect = np.real(np.trace(H*psirho))

   
        # prob = psirho.diag()
        # #prob = generateRandomProb_s(prob, stats = 10000)
        # ez = np.dot(prob, Basez) 
        # ezz = np.dot(prob, Basezz) 

        # gry = qtp.Qobj(tensorl([ry(np.pi / 2, self.noise) for i in range(N)]), dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper', isherm = True)
        # psirhox = gry.dag() * psirho * gry
        # prob = psirhox.diag()
        # #prob = generateRandomProb_s(prob, stats = 10000)
        # exx = np.dot(prob, Basezz) 

        # grx = qtp.Qobj(tensorl([rx(np.pi / 2, self.noise) for i in range(N)]), dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper', isherm = True)
        # psirhoy = grx.dag() * psirho * grx
        # prob = psirhoy.diag()
        # #prob = generateRandomProb_s(prob, stats = 10000)
        # eyy = np.dot(prob, Basezz) 

        # if self.Htype == "XY":
        #     Hexpect = self.h*ez + exx + eyy
        # elif self.Htype == "XXZ":
        #     Hexpect = self.h*ezz + exx + eyy

        return  Hexpect

    def q_expect(self, x, U):
        if self.noise:
            #print("noised") 
            psi = tensorl([spin[int(i)] for i in x])
            psi = np.dot(U, psi)

            prob = np.abs(psi) ** 2
            prob = generateRandomProb_s(prob, stats = 10000)
            ez = np.dot(prob, Basez)
            ezz = np.dot(prob, Basezz) 

            psix = np.dot(tensorl([ry(np.pi / 2, self.noise) for i in range(self.N)]), psi)
            prob = np.abs(psix) ** 2
            prob = generateRandomProb_s(prob, stats = 10000)
            exx = np.dot(prob, Basezz) 

            psiy = np.dot(tensorl([rx(np.pi / 2, self.noise) for i in range(self.N)]), psi)
            prob = np.abs(psiy) ** 2
            prob = generateRandomProb_s(prob, stats = 10000)
            eyy = np.dot(prob, Basezz) 

            if self.Htype == "XY":
                Hexpect = self.h*ez + exx + eyy
            elif self.Htype == "XXZ":
                Hexpect = self.h*ezz + exx + eyy
            
            return  Hexpect
        
        else:
            H = (self.H).H
            psi = tensorl([spin[int(i)] for i in x])
            psi = np.dot(U, psi)

            return  np.real(np.dot(np.conj(psi).T, np.dot(H, psi))/np.dot(np.conj(psi).T, psi))
        
    
    def get_Ex(self, thetal, samples):
        core_number = config.core_number

        if self.symmetry:
            Ul = [self.get_U(thetal, qN = n) for n in range(N + 1)]
            if self.parallel:         
                with Pool(processes = core_number) as pool:
                    E_x = pool.map(partial(self.q_expects, Ul=Ul), samples)
                E_x = np.array(E_x)

            else:
                E_x = np.array([self.q_expects(x, Ul) for x in samples])
        
        elif self.decoherence:
            if self.parallel:   
                with Pool(processes = core_number) as pool:
                    E_x = pool.map(partial(self.q_expect_rho, thetal=thetal), samples)
                E_x = np.array(E_x)
            else:
                E_x = np.array([self.q_expect_rho(x, thetal) for x in samples])
            
        else:
            U = self.get_U(thetal)
            if self.parallel:
                with Pool(processes = core_number) as pool:      
                    E_x = pool.map(partial(self.q_expect, U=U), samples)
                E_x = np.array(E_x)
            else:
                E_x = np.array([self.q_expect(x, U) for x in samples])
        
        return E_x
        

    def loss_quantum(self, thetal, alpha, samples, bound=False):
        if bound:
            phi = alpha
        else:
            phi = np.exp(alpha)/np.sum(np.exp(alpha)) if self.join else 1/(1+np.exp(-alpha))

    
        E_x = self.get_Ex(thetal, samples)
        #print("E_x", np.sort(E_x))
        if self.samp:
            return np.mean(E_x)
        else:
            if self.join:
                return np.dot(phi, E_x)
            else:
                prob = jointprob(samples, phi)
                if self.peierls:
                    prob = np.exp(-self.beta*E_x)
                    prob = prob/np.sum(prob)
                return np.dot(prob, E_x)
    
    def loss_quantum_grad(self, thetal, alpha, samples, gi, a, b):
        phi = np.exp(alpha)/np.sum(np.exp(alpha)) if self.join else 1/(1+np.exp(-alpha))
        
        U = self.get_U_grad(thetal, gi, a, b)
        #q_expectl = [q_expect(U, x, H) for x in samples]
        q_expectl = [self.q_expect(x, U) for x in samples]

        if self.samp: # Using samp
            return np.mean(q_expectl)
        else:
            if self.join:
                return np.dot(phi, q_expectl)
            else:
                prob = jointprob(samples, phi)
                return np.dot(prob, q_expectl)
    
    def loss_func(self, para, samples, bound=False):
        if self.peierls:
            thetal = para
            if self.samp:
                samples = gen_btsl_sub(self.N, self.nbatch)
            E_x = self.get_Ex(thetal, samples)
            loss =  -np.log(np.sum(np.exp(-self.beta*E_x)))
            return loss

        else:
            if self.join:
                Nphi = len(samples)
                phi = para[:Nphi] if bound else np.exp(para[:Nphi])/np.sum(np.exp(para[:Nphi]))
                thetal = para[Nphi:]
                U = self.get_U(thetal)
                loss_samp = np.log(phi) + [self.beta*self.q_expect(x, U) for x in samples]
                return np.dot(phi, loss_samp)

            else:
                phi = para[:self.N] if bound else 1 / (1 + np.exp(-para[:self.N]))
                thetal = para[self.N:]
                E_x = self.get_Ex(thetal, samples)
                logp_x = np.array([logp(phi, x) for x in samples])
                loss_samp = logp_x + self.beta*E_x

                if self.samp:
                    return np.mean(loss_samp)
                else:
                    prob = jointprob(samples, phi)
                    return np.dot(prob, loss_samp)
    
    def grad_parashift_glob(self, thetal, alpha, samples):
        grad_theta = []
        dth = np.zeros(len(thetal))
        for i in range(len(thetal)):
            dth[i] = np.pi/2
            loss1 = self.loss_quantum(thetal + dth, alpha, samples)
            dth[i] = -np.pi/2
            loss1 -= self.loss_quantum(thetal + dth, alpha, samples)
            grad_theta.append(loss1*0.5)
            dth[i] = 0
    
        return np.array(grad_theta)

    def grad_parashift_XY(self, thatal, alpha, samples):
        grad_theta = []
        dth = np.zeros(len(thetal))
        for i in range(len(thetal)):
            if not (i % 4 == 2):
                ###parameter shift for single-qubit gate
                dth[i] = np.pi/2
                loss1 = self.loss_quantum(thetal + dth, alpha, samples)
                dth[i] = -np.pi/2
                loss1 -= self.loss_quantum(thetal + dth, alpha, samples)
                grad_theta.append(loss1*0.5)
                dth[i] = 0

            else:
                ####parameter shift for XY gate
                lossX = self.loss_quantum_grad(thetal, alpha, samples, i, np.pi/8, np.pi/8) - self.loss_quantum_grad(thetal, alpha, samples, i, -np.pi/8, -np.pi/8)
                
                lossY = self.loss_quantum_grad(thetal, alpha, samples, i, np.pi/ 8, -np.pi/8) - self.loss_quantum_grad(thetal, alpha, samples, i, -np.pi/8, np.pi/8)
                grad_theta.append(lossX + lossY)
        
        return np.array(grad_theta)

    def grad(self, para, samples):
        if self.join:
            alpha = para[:int(2**self.N)]
            phi = np.exp(alpha) / np.sum(np.exp(alpha))
            thetal = para[int(2**self.N):]
            U = self.get_U(thetal)
            grad_logp = -np.outer(phi, phi)
            for i in range(len(phi)):
                grad_logp[i, i] = phi[i] - phi[i]**2
            
            fx = np.log(phi) + self.beta * np.array([self.q_expect(x, U) for x in samples])
            grad_phi = (1 / self.beta) * np.dot((1 + fx), grad_logp)

        else: 
            alpha = para[:self.N]
            phi = 1 / (1 + np.exp(-alpha))
            thetal = para[self.N:]
            prob = jointprob(samples, phi)
            b = 0.0
            if self.samp: 
                b = self.loss_func(para, samples)
                prob = np.ones(len(samples))/len(samples)
            
            if self.symmetry:
                Ul = [self.get_U(thetal, qN = n) for n in range(N + 1)] 
            else:
                U = self.get_U(thetal)

            grad_phil = []
            for x in samples:
                grad_logp = (x - phi)
                if self.symmetry:
                    fx = logp(phi, x) + self.beta*self.q_expects(x, Ul)
                else:
                    fx = logp(phi, x) + self.beta*self.q_expect(x, U) 
                grad_phil.append((fx - b)*grad_logp)
            grad_phi = np.dot(prob, grad_phil)


        #grad_theta1 = scipy.optimize.approx_fprime(thetal, self.loss_quantum, 1e-8, alpha,  samples)
        grad_theta = self.grad_parashift_glob(thetal, alpha, samples)
        #print("difference: ", grad_theta1-grad_theta)           
        grad = np.concatenate((grad_phi, self.beta * grad_theta))

        return grad            

    def optimize_adam(self, samples):
        np.random.seed()
        if self.gate == "XY":
            thetal = 2*np.pi*np.random.rand(self.layer_number*(self.N-1)*4)
        elif self.gate == "chainXY" or self.gate == "fcXY":
            thetal = 2*np.pi*np.random.rand(self.layer_number*(self.N))

        if self.join:
            a = np.zeros(len(samples))
            phi = np.exp(a) / np.sum(np.exp(a))
            para = np.concatenate((a, thetal))
        else:
            a = np.zeros(N)
            phi = 1 / (1 + np.exp(-a))
            para = np.concatenate((a, thetal))
        
        
        if self.peierls:
            para = thetal
        
        b1 = 0.9
        b2 = 0.999
        e = 0.00000001
        mt = np.zeros(len(para))
        vt = np.zeros(len(para))        
        lossfl = []
        paral = []
        for i in range(self.niter):
            print("Current interation: %d/%d" % (i+1, self.niter))
            if self.samp:
                samples = gen_samples(phi, self.nbatch, N)
                #samples = gen_btsl_sub(N, int(2**N))
            
            # totalbasis = gen_btsl(N)
            # samples = gen_btsl_sub(N, 16)

            lossf = self.loss_func(para, samples) / self.beta
            #lossft = loss_func(para, totalbasis, beta, H, layer_number,  nbatch = nbatch, gate = gate, samp = samp, join = join, noise = noise, symmetry = symmetry)
            pid = os.getpid()
            print("process: ", pid, "Current loss: ", lossf)
            lossfl.append(lossf)
            paral.append(para)
        
            grads = self.grad(para, samples)
            mt = b1*mt + (1-b1)*grads
            vt = b2*vt + (1-b2)*grads**2
            mtt = mt/(1-b1**(i+1))
            vtt = vt/(1-b2**(i+1))
            lr = self.lr
            ###learning rate decay####
            if i > 20:
                print("decay")
                lr = self.decay

            para = para - lr * mtt / (np.sqrt(vtt) + e)

            if self.join:
                Nphi = len(samples)
                phi = np.exp(para[: Nphi])/np.sum(np.exp(para[:Nphi]))
            else:
                phi = 1/(1+np.exp(-para[:self.N]))
                thetal = para[self.N:]

        return para, np.array(paral), np.array(lossfl)
        
    def optimize_zoopt(self, samples):
        np.random.seed()
        lossfl = []
        def objfunction(solution):
            x = solution.get_x()
            value = self.loss_func(np.array(x), samples, bound=True)
            return value

        dim_theta = self.layer_number * (self.N)
        dim = N + dim_theta
        obj = Objective(objfunction, Dimension(dim, [[0, 1]]*self.N + [[0, 2*np.pi]]*(dim_theta), [True]*dim))

        #peierls
        if self.peierls:
            dim = dim_theta
            obj = Objective(objfunction, Dimension(dim, [[0, 2*np.pi]]*(dim_theta), [True]*dim))

        parameter = Parameter(budget=10000, exploration_rate=0.02, parallel=True, server_num=6)
        sol = Opt.min(obj, parameter)
        para = np.array(sol.get_x())
        lossfl = np.array(obj.get_history_bestsofar()) / self.beta
        
        return para, lossfl
    
    def opt(self, ti):
        if self.optimizer == "adam":
            bound = False
            para_opt, paral, lossfl = self.optimize_adam(self.var_basis)
        elif self.optimizer == "zoopt":
            bound = True
            para_opt, lossfl = self.optimize_zoopt(self.var_basis)

        if not os.path.exists("data"):
                os.mkdir("data")
            
        self.sol = para_opt
        self.traj = lossfl
        
        F_sol, E_sol = self.cal_obs(self.sol, bound=bound)
        F = exact(self.beta, (self.H).H)[0]
        E = exact(self.beta, (self.H).H)[1]
        print("F: ", F, " ", "F_opt: ", F_sol, "\n")
        print("E: ", E, " ", "E_opt: ", E_sol, "\n")

        tag = self.create_tag()
        tag_traj = "data/" + tag + "_ti" + str(ti)
        tag_sol = "data/" + "sol_" + tag + "_ti" + str(ti)
        
        np.save(tag_traj, self.traj)
        np.save(tag_sol, self.sol)
        
        if self.optimizer == "adam":
            self.xtraj = paral
            tag_xtraj = "data/" + "xtraj_" + tag + "_ti" + str(ti)
            np.save(tag_xtraj, self.xtraj)   


        return self.sol, self.traj 


    def learn(self, trialnum=1, parallel_trial=False):
        if parallel_trial:
             with Pool(processes = trialnum) as pool:
                pool.map(self.opt, range(trialnum)) 
        else:
            for ti in range(trialnum):
                self.opt(ti)
                    

    def cal_obs(self, x, bound=False):
        samples = self.var_basis
        if self.join:
            alpha = x[:len(self.var_basis)]
            theta=  x[len(self.var_basis):]
        else:
            alpha = x[:self.N]
            theta = x[self.N:]
        
        if bound:
            phi = alpha
        else:
            phi = 1 / (1+np.exp(-alpha))

        if self.samp:
            samples=  gen_samples(phi, 1000, self.N)
        
        if self.peierls:
            theta = x

        F = self.loss_func(x, samples, bound=bound) / self.beta
        E = self.loss_quantum(theta, alpha, samples, bound=bound)

        return F, E



if __name__ == "__main__":
    N = config.N
    beta = 0.2
    h = 0.5
    layer_number = config.layer_number
    trialnum = 4
    #bVQE = betaVQE(N, beta, h, layer_number, optimizer="adam", samp=False, decoherence=True, parallel=False)
    #bVQE.learn(trialnum=trialnum, parallel_trial=False)
    # F = exact(bVQE.beta, (bVQE.H).H)[0]


    bVQE = betaVQE(N, beta, h, layer_number, optimizer="adam", samp=False,  parallel=False)
    tag = bVQE.create_tag()
    tag_sol = "data/" + "sol_" + tag + "_ti" + str(2) + ".npy"
    sol = np.load(tag_sol)
    theta = sol[N:]
    samples = gen_btsl(N)
    print("------------Decoherence optimized results-------")
    bVQE.decoherence = True
    F, E = bVQE.cal_obs(sol)
    E_x = bVQE.get_Ex(theta, samples)
    print(np.sort(E_x))
    print("F: ", F, " ", "E: ", E)

    print("------------No Decoherence optimized results-------")
    bVQE.decoherence = False
    F, E = bVQE.cal_obs(sol)
    E_x = bVQE.get_Ex(theta, samples)
    print(np.sort(E_x))
    print("F: ", F, " ", "E: ", E)
    
    print("------------Exact results--------------------")
    H = (bVQE.H).H
    F_exact = exact(beta, H)[0]
    E_exact = exact(beta, H)[1]
    print(np.linalg.eigvalsh(H))
    print("F_exact: ", F_exact, ' ', "E_exact: ", E_exact)

    # fig, ax = plt.subplots()
    # for ti in range(trialnum):
    #     tagi = "data/" + tag + "_ti" + str(ti)
    #     opt_traj = np.load(tagi + ".npy")
    #     ax.plot(np.arange(0, len(opt_traj)), opt_traj) 
    #     ax.plot([0, len(opt_traj)], [F, F], '--')
    #     ax.set_xlabel("budget", fontsize = 12)
    #     ax.set_ylabel("loss function", fontsize = 12)
    #     ax.text(0, 1.05, tag, wrap = True, transform = ax.transAxes, fontsize = 8)

    # if not os.path.exists("results"):
    #         os.mkdir("results")
    # tagfig = "results/" + tag + ".png" 
    # fig.savefig(tagfig, dpi = 400)
        




    

