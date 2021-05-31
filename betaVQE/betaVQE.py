import numpy as np
from func import *
import scipy.optimize
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
from functools import partial
#from scipydirect import minimize
import os
# from zoopt import Dimension, Objective, Parameter, Opt, Solution, ExpOpt
#from gpso import minimize
import config
from multiprocessing import Pool
#import scipydirect
import time 

class betaVQE():
    def __init__(self, N, beta, h, layer_number, delta=0.5, niter=150, nbatch=10, Htype="XY", gate="chainXY", optimizer="adam", samp=False, join=False, noise=False, decay=0.05, lr=0.1, symmetry=False, peierls=False, decoherence=False, state2=False, parallel=False, spsa_gradient=False, spsa_iter=10, xx=False, savedir="expdata/", decay_step=20):
        ''' Initial function'''
        self.N = N
        self.beta = beta
        self.h = h
        self.delta = delta
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
        self.state2 = state2
        self.parallel = parallel
        self.H = Hamiltonian(h=self.h, N=self.N, delta = self.delta, Htype = self.Htype)
        self.var_basis = gen_btsl(N)
        self.sol = []
        self.traj = []
        self.xtraj = []
        self.spsa_gradient = spsa_gradient
        self.spsa_iter = spsa_iter
        self.xx = xx
        self.savedir=savedir
        self.decay_step = decay_step

    def create_tag(self):
        tag = "Hamil-" + str(self.Htype)
        tag += "_N" + str(self.N)
        tag += "_beta" + str(self.beta)
        tag += "_h" + str(self.h)
        if self.Htype == "XXZ":
            tag += "_delta" + str(self.delta)
        tag += "_layer_number" + str(self.layer_number)
        tag += "_method-" + str(self.optimizer)
        tag += "_spgrad" if self.spsa_gradient else ""
        tag += "_gate-" + str(self.gate)
        tag += "_noise" if self.noise else ""
        tag += "_sym" if self.symmetry else ""
        tag += "_peierls" if self.peierls else ""
        tag += "_decoherence" if self.decoherence else ""
        tag += "_state2" if self.state2 else ""
        tag += "_join" if self.join else ""
        tag += "_xx" if self.xx else ""
        if not self.niter == 150:
            tag += "_niter%d" %self.niter

        #tag += "_10ns"
        

        if self.samp:
            tag += "_nbatch" + str(self.nbatch)
        
        if self.optimizer == "adam":
            tag += "_lr" + str(self.lr)
            tag += "_decay" + str(self.decay)
        
        tag += "_%.3fpi" %(angle/np.pi)
        
        return tag

    def cal_H(self):
        self.H = Hamiltonian(h=self.h, N=self.N, delta=self.delta, Htype = self.Htype)
        return self.H.H

        
    def get_U(self, thetal, qN=None):
        if self.gate == "XY":
            theta = thetal.reshape((self.layer_number, self.N - 1, 4))
            U = 1.0
            for l in range(theta.shape[0]): # layer
                Ul = 1.0
                for i in range(theta.shape[1]):
                    oplist = [si for n in range(N - 1)] 
                    oplist[i] = enhan_XYgate(theta[l, i, :], noise=self.noise, dthedphi=ERROR_XY_LIST[i])
                    tq_gateN = tensorl(oplist)
                    Ul = np.dot(tq_gateN , Ul)
                U = np.dot(Ul, U)
        
        elif self.gate == "chainXY":
            #simulation
            theta = thetal.reshape((self.layer_number, self.N))
            U = 1.0
            for l in range(theta.shape[0]):
                Ul = ChainXY
                Ul = np.dot(singleZlayer(theta[l]), Ul)
                U = np.dot(Ul, U)
            # U  = np.dot(chain_XY(angle, N, qN=qN), U)

            # ##experimental simultion 
            # theta = thetal.reshape((self.layer_number, self.N))
            # U = 1.0
            # for l in range(theta.shape[0]):
            #     U = Int_layer.dot(U)
            #     singU = rz_layer(theta[l])
            #     U = singU.dot(U)
            # #U = Int_layer.dot(U)
            # U = U_IDLE.dot(U)


        # elif self.gate == "fcXY":
        #     theta = thetal.reshape((self.layer_number, self.N))
        #     U = 1.0
        #     for l in range(theta.shape[0]):
        #         Ul = singlelayer(theta[l, :N], N, "Z", noise=self.noise, qN = qN)
        #         Ul = np.dot(fc_XY(np.pi / 8, N, qN = qN), Ul)
        #         U = np.dot(Ul, U)
        
        return U

   

    def q_expect(self, x, U):
        stats = 2000
        if self.noise:
            #print("noised") 
            psi = tensorl([spin[int(i)] for i in x])
            psi = np.dot(U, psi)

            #Global observables
            prob = np.abs(psi) ** 2
            prob = generateRandomProb_s(prob, stats = stats)
            ez = np.dot(prob, Basez)
            ezz = np.dot(prob, Basezz) 

            psix = np.dot(tensorl([ry(np.pi / 2, noise=self.noise) for i in range(self.N)]), psi)
            prob = np.abs(psix) ** 2
            prob = generateRandomProb_s(prob, stats = stats)
            exx = np.dot(prob, Basezz) 

            psiy = np.dot(tensorl([rx(np.pi / 2, noise=self.noise) for i in range(self.N)]), psi)
            prob = np.abs(psiy) ** 2
            prob = generateRandomProb_s(prob, stats = stats)
            eyy = np.dot(prob, Basezz) 


            # ##Local observables
            # prob = np.abs(psi) ** 2
            # prob = generateRandomProb_s(prob, stats = stats)
            # ez = 0
            # ezz = 0
            # for i in range(N):
            #     bits = [i]
            #     ez += measure_obs(self.N, prob, bits)
            #     if i < N-1:
            #         bits = [i, i+1]
            #         ezz += measure_obs(self.N, prob, bits)

            # psix = np.dot(tensorl([ry(np.pi / 2, noise=self.noise) for i in range(self.N)]), psi)
            # prob = np.abs(psix) ** 2
            # prob = generateRandomProb_s(prob, stats = stats)
            # exx = 0
            # for i in range(N-1):
            #     bits = [i, i+1]
            #     exx += measure_obs(self.N, prob, bits)

            # psiy = np.dot(tensorl([rx(np.pi / 2, noise=self.noise) for i in range(self.N)]), psi)
            # prob = np.abs(psiy) ** 2
            # prob = generateRandomProb_s(prob, stats = stats)
            # eyy = 0
            # for i in range(N-1):
            #     bits = [i, i+1]
            #     eyy += measure_obs(self.N, prob, bits)
        else:
            H = (self.H).H
            psi = tensorl([spin[int(i)] for i in x])
            psi = np.dot(U, psi)

            return  np.real(np.dot(np.conj(psi).T, np.dot(H, psi))/np.dot(np.conj(psi).T, psi))
        
    def q_expect2(self, x, Rz_layers):
        stats = 1000
        N = self.N
        if self.state2:          
            psi_idle = self.get_Upsi(x, Rz_layers)
            prob = cut_psi(psi_idle.data.toarray().ravel())
            if self.noise:
                prob = generateRandomProb_s(prob, stats=stats)
            expZ = prob.dot(baseZ3)
            expZZ = prob.dot(baseZZ3)

            psi_idlex = Ry_pid2 * psi_idle
            probx = cut_psi(psi_idlex.data.toarray().ravel())
            if self.noise:
                probx = generateRandomProb_s(probx, stats=stats)
            expXX = probx.dot(baseZZ3)

            psi_idley = Rx_pid2 * psi_idle
            proby = cut_psi(psi_idley.data.toarray().ravel())
            if self.noise:
                proby = generateRandomProb_s(proby, stats=stats)
            expYY = proby.dot(baseZZ3)

            if self.Htype == "XY":
                if self.xx:
                    Hexpect = np.real(self.h*expZ + 2*expXX)
                else:
                    Hexpect = np.real(self.h*expZ + expXX + expYY)
            elif self.Htype == "XXZ":
                Hexpect = np.real(self.delta*expZZ + expXX + expYY - self.h*expZ)

            # elif self.Htype == "XXZ":
            #     Hexpect = np.real(self.h*expZZ + 2*expXX)
            
            return  Hexpect
        
        else:
            psi_idle = self.get_Upsi(x, Rz_layers)
            return  np.real(np.dot(np.conj(psi_idle).T, np.dot(self.H.H, psi_idle)))
    
    def get_Upsi(self, x, Rz_layers):
        if self.state2:
            psi = get_psi(x)
            for i in range(self.layer_number):
                psi = Int_layer2 * (psi)
                psi = Rz_layers[i] * (psi)
                
            psi_idle = U_IDLE2 * (psi)
        
        else: 
            # psi = tensorl([spin[int(i)] for i in x])
            # for i in range(self.layer_number):
            #     psi = Int_layer.dot(psi)
            #     psi = Rz_layers[i].dot(psi)
            # psi_idle = U_IDLE.dot(psi)

            ##Using exact gate
            psi = tensorl([spin[int(i)] for i in x])
            for i in range(self.layer_number):
                psi = ChainXY.dot(psi)
                psi = Rz_layers[i].dot(psi)
                
            psi_idle = psi

        return psi_idle

    def expect_corr(self, x, Rz_layers):
        stats = 200
        N = self.N
        ZZ_corr = []    
        if self.state2:
            psi_idle = self.get_Upsi(x, Rz_layers)
            prob = cut_psi(psi_idle.data.toarray().ravel())
            if self.noise:
                prob = generateRandomProb_s(prob, stats=stats)
            
            ZZ_corr = []
            for i in range(1, N):
                bits = [0, i]
                ezz = measure_obs(self.N, prob, bits)   
                ZZ_corr.append(ezz)
        else:
            psi_idle = self.get_Upsi(x, Rz_layers)
            ZZ_corr = []
            for i in range(1, N):
               ## ---- direct calculation-----
                
                zz = Sz[0].dot(Sz[i])
                ZZ_corr.append(np.real(np.dot(np.conj(psi_idle).T, np.dot(zz, psi_idle))))

                # #--------measure-----------
                # prob = np.abs(psi_idle) ** 2
                # bits = [0, i]
                # ezz = measure_obs(self.N, prob, bits)
                # ZZ_corr.append(ezz)

        return ZZ_corr

    def get_corr(self, para, samples):
        core_number = config.core_number
        thetal = para[self.N:]
        phi = 1 / (1 + np.exp(-para[:self.N]))
        thetal[thetal > np.pi] -= 2 * np.pi
        thetal[thetal < -np.pi] += 2 * np.pi

        theta = thetal.reshape(self.layer_number, self.N)
    
        if self.state2:
            Rz_layersf = rz_layer2
        else:
            Rz_layersf = rz_layer
        
        Rz_layers =  [Rz_layersf(theta[i]) for i in range(self.layer_number)]
        corr = np.array([self.expect_corr(x, Rz_layers) for x in samples])

        if self.samp:
            return np.mean(corr, axis=0)
        else:
            prob = jointprob(samples, phi)
            prob = prob.reshape(len(prob), 1)
            return np.sum(prob*corr, axis=0)



    def get_Ex(self, thetal, samples):
    
        core_number = config.core_number
        thetal[thetal > np.pi] -= 2 * np.pi
        thetal[thetal < -np.pi] += 2 * np.pi

        theta = thetal.reshape(self.layer_number, self.N)
        if self.symmetry:
            Ul = [self.get_U(thetal, qN = n) for n in range(N + 1)]
            if self.parallel:         
                with Pool(processes = core_number) as pool:
                    Ex = pool.map(partial(self.q_expects, Ul=Ul), samples)
                Ex = np.array(Ex)

            else:
                Ex = np.array([self.q_expects(x, Ul) for x in samples])
        
        elif self.decoherence:
            if self.parallel:   
                with Pool(processes = core_number) as pool:
                    Ex = pool.map(partial(self.q_expect_rho, thetal=thetal), samples)
                Ex = np.array(Ex)
            else:
                Ex = np.array([self.q_expect_rho(x, thetal) for x in samples])

        ##Note that all x use the same rz layer 
        else:
            # #U = self.get_U(thetal)
            # if self.parallel:
            #     with Pool(processes = core_number) as pool:      
            #         Ex = pool.map(partial(self.q_expect, U=U), samples)
            #     Ex = np.array(Ex)
            # else:
            #     Ex = np.array([self.q_expect(x, U) for x in samples])

            
            # if self.state2:
            #     Rz_layers = [rz_layer2(theta[i]) for i in range(self.layer_number)]
            # else:
            #     Rz_layers = [rz_layer(theta[i]) for i in range(self.layer_number)]

            if self.state2:
                Rz_layersf = rz_layer2
            else:
                Rz_layersf = rz_layer
                #Using exact gate
                Rz_layersf = singleZlayer

            # if self.noise:
            #     Ex = []
            #     for x in samples:
            #         Rz_layers = [Rz_layersf(theta[i]+np.random.uniform(-0.04, 0.04, len(theta[i]))) for i in range(self.layer_number)]
            #         Ex.append(self.q_expect2(x, Rz_layers))
            #     Ex = np.array(Ex)

            
            Rz_layers =  [Rz_layersf(theta[i]) for i in range(self.layer_number)]

            
            Ex = np.array([self.q_expect2(x, Rz_layers) for x in samples])

        return Ex


    def loss_quantum(self, thetal, alpha, samples, bound=False, Ex=[]):
        if bound:
            phi = alpha
        else:
            phi = np.exp(alpha)/np.sum(np.exp(alpha)) if self.join else 1/(1+np.exp(-alpha))

        if len(Ex) == 0:
            Ex = self.get_Ex(thetal, samples)
        if self.samp:
            return np.mean(Ex)
            # #use prob
            # prob = jointprob(samples, phi)
            # return np.dot(prob, Ex)

        else:
            if self.join:
                return np.dot(phi, Ex)
            else:
                prob = jointprob(samples, phi)
                if self.peierls:
                    prob = np.exp(-self.beta*Ex)
                    prob = prob/np.sum(prob)
                return np.dot(prob, Ex)
    
   

    def loss_func(self, para, samples, bound=False, Ex=[]):
        if self.peierls:
            thetal = para
            if self.samp:
                samples = gen_btsl_sub(self.N, self.nbatch)
            if len(Ex) == 0:
                Ex = self.get_Ex(thetal, samples)
            loss =  -np.log(np.sum(np.exp(-self.beta*Ex)))
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
                if len(Ex) == 0:
                    Ex = self.get_Ex(thetal, samples)
                logp_x = np.array([logp(phi, x) for x in samples])
                loss_samp = logp_x + self.beta*Ex

                if self.samp:
                    return np.mean(loss_samp)
                    # prob = jointprob(samples, phi)
                    # return np.dot(prob, Ex)

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
    
    def grad_spsa(self, thetal, alpha, samples, n_iter,c_par = 0.25, gamma=1/6):
            ck = c_par/( n_iter + 1 )**gamma
            num_p=len(thetal)
            delta = (np.random.randint(0, 2, num_p) * 2 - 1)
            ghat = 0.  # Initialise gradient estimate
            for j in np.arange (self.spsa_iter):
                # This loop produces ``ens_size`` realisations of the gradient
                # which will be averaged. Each has a cost of two function runs.
                # Bernoulli distribution with p=0.5
                delta = (np.random.randint(0, 2, num_p) * 2 - 1)
                # Stochastic perturbation, innit
                theta_plus = thetal + ck*delta
                theta_minus = thetal - ck*delta
                # Funcion values associated with ``theta_plus`` and 
                #``theta_minus``
                j_plus = self.loss_quantum(theta_plus, alpha, samples)
                j_minus = self.loss_quantum(theta_minus, alpha, samples)
                # Estimate the gradient
                ghat = ghat + ( j_plus - j_minus)/(2.*ck*delta)
            # Average gradient...
            ghat = ghat/float(self.spsa_iter)
            return ghat

    def grad(self, para, samples, n_iter,  Ex=[]):
        if self.join:
            alpha = para[:int(2**self.N)]
            phi = np.exp(alpha) / np.sum(np.exp(alpha))
            thetal = para[int(2**self.N):]
            U = self.get_U(thetal)
            grad_logp = -np.outer(phi, phi)
            for i in range(len(phi)):
                grad_logp[i, i] = phi[i] - phi[i]**2
            
            fx = np.log(phi) + self.beta * np.array([self.q_expect(x, U) for x in samples])
            grad_phi = np.dot((1 + fx), grad_logp)

        else: 
            alpha = para[:self.N]
            phi = 1 / (1 + np.exp(-alpha))
            thetal = para[self.N:]
            prob = jointprob(samples, phi)
            b = 0.0
            if self.samp: 
                b = self.loss_func(para, samples, Ex=Ex)
                prob = np.ones(len(samples))/len(samples)
                # prob = jointprob(samples, phi)
            
            # if self.symmetry:
            #     Ul = [self.get_U(thetal, qN = n) for n in range(N + 1)] 
            # else:
            #     U = self.get_U(thetal)

            grad_phil = []
            
            if len(Ex) == 0:
                Ex = self.get_Ex(thetal, samples)

            grad_logp_x = []
            logp_x = []
            for x in samples:
                grad_logp_x.append(x - phi)
                logp_x.append(logp(phi, x))

            fx = np.array(logp_x) + self.beta*Ex
            #fx *= 10
            grad_phil = (fx-b).reshape(len(fx), 1) * np.array(grad_logp_x)

            # for x in samples:
            #     grad_logp = (x - phi)
            #     if self.symmetry: 
            #         fx = logp(phi, x) + self.beta*self.q_expects(x, Ul)
            #     else:
            #         fx = logp(phi, x) + self.beta*self.q_expect(x, U) 
            #     grad_phil.append((fx - b)*grad_logp)

            grad_phi = np.dot(prob, grad_phil)


        #grad_theta1 = scipy.optimize.approx_fprime(thetal, self.loss_quantum, 1e-8, alpha,  samples)
        if self.spsa_gradient:
            grad_theta = self.grad_spsa(thetal, alpha, samples, n_iter)
        else:
            grad_theta = self.grad_parashift_glob(thetal, alpha, samples) #Need new Ex
        #print("difference: ", grad_theta1-grad_theta)           
        grad = np.concatenate((grad_phi, self.beta * grad_theta))

        return grad            

    def optimize_adam(self, samples, ti):
        btsl = gen_btsl(self.N)
        if not os.path.exists(self.savedir):
                os.mkdir(self.savedir)
            
        np.random.seed()
        if self.gate == "XY":
            thetal = 2*np.pi*np.random.rand(self.layer_number*(self.N-1)*4)
        elif self.gate == "chainXY" or self.gate == "fcXY":
            thetal = 2*np.pi*np.random.rand(self.layer_number*(self.N)) - np.pi

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
        
        phi0 = phi.copy()
        b1 = 0.9
        b2 = 0.999
        e = 0.00000001
        mt = np.zeros(len(para))
        vt = np.zeros(len(para))        
        lossfl = []
        paral = []
        loss32l = []
        Ex32l = []
   
            
        for i in range(self.niter):

            print("Current interation: %d/%d" % (i+1, self.niter))
            if self.samp:
                samples = gen_samples(phi, self.nbatch, N)
                # #use uniform sampels
                # samples = gen_samples(phi0, self.nbatch, N)
                #samples = gen_btsl_sub(N, int(2**N))
            
            # totalbasis = gen_btsl(N)
            # samples = gen_btsl_sub(N, 16)
            # if self.noise:
            #     Ex = self.get_Ex(thetal+rz_err_theta, samples)
            #     lossf = self.loss_func(para+rz_err_para, samples, Ex=Ex) / self.beta
            # else:
            Ex = self.get_Ex(thetal, samples)
            lossf = self.loss_func(para, samples, Ex=Ex) / self.beta
            lossfl.append(lossf)
            paral.append(para)
            #------- calculate unbaised quantity-------
            if self.samp:
                self.samp = False
                Ex32 = self.get_Ex(thetal, btsl)
                loss32 = self.loss_func(para, btsl, Ex=Ex32)/ self.beta 
                loss32l.append(loss32)
                Ex32l.append(Ex32)
                self.samp=True

            #lossft = loss_func(para, totalbasis, beta, H, layer_number,  nbatch = nbatch, gate = gate, samp = samp, join = join, noise = noise, symmetry = symmetry)
            pid = os.getpid()
            print("process: ", pid, "Current loss: ", lossf)
    
           
            # if self.noise:
            #     grads = self.grad(para+rz_err_para, samples, i,  Ex=Ex)
            #else:
            grads = self.grad(para, samples, i,  Ex=Ex)

            #print(grads)
            mt = b1*mt + (1-b1)*grads
            vt = b2*vt + (1-b2)*grads**2

            
            mtt = mt/(1-b1**(i+1))
            vtt = vt/(1-b2**(i+1))
            lr = self.lr

            ###learning rate decay####
            if i > self.decay_step:
                print("decay")
                lr = self.decay

            print("step length:", lr * mtt / (np.sqrt(vtt) + e))
            para = para - lr * mtt / (np.sqrt(vtt) + e)
            #constrain theta to interval [-pi, pi]
            para[self.N:][para[self.N:] > np.pi ] -= 2 * np.pi
            para[self.N:][para[self.N:] < -np.pi ] += 2 * np.pi

            if self.join:
                Nphi = len(samples)
                phi = np.exp(para[: Nphi])/np.sum(np.exp(para[:Nphi]))
            else:
                phi = 1/(1+np.exp(-para[:self.N]))
                thetal = para[self.N:]

            if i == (self.niter-1):
                Ex = self.get_Ex(thetal, samples)
                lossf = self.loss_func(para, samples, Ex=Ex) / self.beta
                lossfl.append(lossf)
                paral.append(para)
                #------- calculate unbaised quantity-------
                if self.samp:
                    self.samp = False
                    Ex32 = self.get_Ex(thetal, btsl)
                    loss32 = self.loss_func(para, btsl, Ex=Ex32)/ self.beta 
                    loss32l.append(loss32)
                    Ex32l.append(Ex32)
                    self.samp=True

        tag = self.create_tag()
        tag_traj = self.savedir + tag + "_ti" + str(ti)
        tag_sol = self.savedir + "sol_" + tag + "_ti" + str(ti)
        tag_loss32 = self.savedir + "loss32_" + tag + "_ti" + str(ti)
        tag_Ex32 = self.savedir + "Ex32_" + tag + "_ti" + str(ti)
        tag_xtraj = self.savedir + "xtraj_" + tag + "_ti" + str(ti)
        
        np.save(tag_loss32, loss32l)
        np.save(tag_Ex32, Ex32l)
        np.save(tag_traj, lossfl)
        np.save(tag_sol, para)
        np.save(tag_xtraj, paral)


                # print(thetal)
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
            para_opt, paral, lossfl = self.optimize_adam(self.var_basis, ti)
        elif self.optimizer == "zoopt":
            bound = True
            para_opt, lossfl = self.optimize_zoopt(self.var_basis)

        self.sol = para_opt
        self.traj = lossfl
        
        F_sol, E_sol, S_sol = self.cal_obs(self.sol, bound=bound)
        F = exact(self.beta, (self.H).H)[0]
        E = exact(self.beta, (self.H).H)[1]
        print("F: ", F, " ", "F_opt: ", F_sol, "\n")
        print("E: ", E, " ", "E_opt: ", E_sol, "\n")

        # tag = self.create_tag()
        # tag_traj = "expdata/" + tag + "_ti" + str(ti)
        # tag_sol = "expdata/" + "sol_" + tag + "_ti" + str(ti)
        # tag_loss32 = "expdata/loss32" + tag + "_ti" + str(ti)
        # tag_Ex32 = "expdata/Ex32" + tag + "_ti" + str(ti)
        # np.save(tag_traj, self.traj)
        # np.save(tag_sol, self.sol)
        
        # if self.optimizer == "adam":
        #     self.xtraj = paral
        #     tag_xtraj = "expdata/" + "xtraj_" + tag + "_ti" + str(ti)
        #     np.save(tag_xtraj, self.xtraj)   


        return self.sol, self.traj 


    def learn(self, trialnum=1, parallel_trial=False, ti=0):
        # if parallel_trial:
        #      with Pool(processes = trialnum) as pool:
        #         pool.map(self.opt, range(trialnum)) 
        # else:
        #     for ti in range(trialnum):
        #         self.opt(ti)
        self.opt(ti)

                    

    def cal_obs(self, x, bound=False, samp_num=20, Ex=[], xsamp=[], thermE=False):
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
            if len(xsamp) == 0:
                samples = gen_samples(phi, samp_num, self.N)
            else:
                samples = xsamp

        if self.peierls:
            theta = x
        
        if len(Ex) == 0:
            Ex = self.get_Ex(theta, samples)
        
        F = self.loss_func(x, samples, bound=bound, Ex=Ex) / self.beta
        E = self.loss_quantum(theta, alpha, samples, bound=bound,Ex=Ex)
        
        prob = jointprob(self.var_basis, phi)
        logpx = np.array([logp(phi, x) for x in self.var_basis])
        S = -1.*np.dot(prob, logpx)
        if thermE:
            E = F + S/self.beta

        return F, E, S

    
    # ##-----------Not used for 2-bit gate verison---------
    # #-------q_expect2---------------------------------------------

                # if ndarray:
            #     psi_idle = self.get_Upis(x, Rz_layers)
            #     prob = cut_psi(psi_idle)
            #     if self.noise:
            #         prob = generateRandomProb_s(prob, stats=stats)
            #     expZ = prob.dot(baseZ3)
            #     expZZ = prob.dot(baseZZ3)

            #     psi_idlex= Ry_pid2.dot(psi_idle)
            #     probx = cut_psi(psi_idlex)
            #     if self.noise:
            #         probx = generateRandomProb_s(probx, stats=stats)
            #     expXX = probx.dot(baseZZ3)

            #     psi_idley = Rx_pid2.dot(psi_idle)
            #     proby = cut_psi(psi_idley)
            #     if self.noise:
            #         proby = generateRandomProb_s(proby, stats=stats)
            #     expYY = proby.dot(baseZZ3)

            # else:

    #  def loss_quantum_grad(self, thetal, alpha, samples, gi, a, b):
    #     '''For XY gate gradient '''
    #     phi = np.exp(alpha)/np.sum(np.exp(alpha)) if self.join else 1/(1+np.exp(-alpha))
        
    #     U = self.get_U_grad(thetal, gi, a, b)
    #     #q_expectl = [q_expect(U, x, H) for x in samples]
    #     q_expectl = [self.q_expect(x, U) for x in samples]

    #     if self.samp: # Using samp
    #         return np.mean(q_expectl)
    #     else:
    #         if self.join:
    #             return np.dot(phi, q_expectl)
    #         else:
    #             prob = jointprob(samples, phi)
    #             return np.dot(prob, q_expectl)
    
    # def get_U_grad(self, thetal, gi, a, b):
    #     '''Only for XY circuit '''
    #     theta = thetal.reshape((self.layer_number, self.N - 1, 4))
    #     layer_ind = gi // ((self.N - 1) * 4)
    #     bit_ind = gi // 4%(self.N - 1)

    #     U = Id
    #     for l in range(theta.shape[0]):
    #         Ul = Id
    #         for i in range(theta.shape[1]):
    #             oplist = [si for n in range(self.N-1)]
    #             if (l == layer_ind) and (i == bit_ind):
    #                 oplist[i] = XYgate_grad(theta[l, i, :], a, b, noise=self.noise, dthedphi=ERROR_XY_LIST[i])
    #             else:
    #                 oplist[i] = enhan_XYgate(theta[l, i, :], noise=self.noise,
    #                 dthedphi=ERROR_XY_LIST[i])

    #             Ul = np.dot(tensorl(oplist), Ul)
    #         U = np.dot(Ul, U)
        
    #     return U

     # def evolve_rho(self, rho, thetal):
    #     theta = thetal.reshape((self.layer_number, self.N))
        
    #     for l in range(theta.shape[0]):
    #         sing_layer =  singlelayer(theta[l, :N], N, "Z", self.noise)
    #         sing_layer = qtp.Qobj(sing_layer, dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper')
    #         vrho = qtp.operator_to_vector(rho)
    #         vrho = Uprop * vrho
    #         rho = qtp.vector_to_operator(vrho)
    #         rho = sing_layer * rho * sing_layer.dag()

    #     return rho

    # def q_expects(self, x, Ul):
    #     H = (self.H).H
    #     #Ul is list contain all qN block
    #     qN = int(np.sum(x))
    #     U = Ul[qN]
    #     psi = tensorl([spin[int(i)] for i in x])
    #     psi = get_block(psi, qN, BasisBlockInd)
    #     H = get_block(H, qN, BasisBlockInd)
    #     psi = np.dot(U, psi)

    #     return np.real(np.dot(np.conj(psi).T, np.dot(H, psi))/np.dot(np.conj(psi).T, psi))

    # def q_expect_rho(self, x, thetal):
    #     stats = 2000
    #     N = self.N
    #     #psi = qtp.tensor([qtp.basis(2, int(i)) for i in x])
    #     psi = qtp.Qobj(tensorl([spin[int(i)] for i in x]), dims = [[2]*N, [1]*N], shape = (2**N, 1), type='ket')
    #     psirho = qtp.ket2dm(psi)
    #     psirho = self.evolve_rho(psirho, thetal)
        
    #     if not self.noise:
    #         H = qtp.Qobj((self.H).H,  dims = [[2]*N, [2]*N], shape = (2**N, 2**N), type='oper', isherm=True)
    #         Hexpect = np.real(np.trace(H*psirho))

    #     else:
    #         prob = psirho.diag()
    #         prob = generateRandomProb_s(prob, stats = stats)
    #         ez = np.dot(prob, Basez) 
    #         ezz = np.dot(prob, Basezz) 

    #         gry = qtp.Qobj(tensorl([ry(np.pi / 2, noise=self.noise) for i in range(N)]), dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper', isherm = True)
    #         psirhox = gry * psirho * gry.dag()
    #         prob = psirhox.diag()
    #         prob = generateRandomProb_s(prob, stats = stats)
    #         exx = np.do+t(prob, Basezz) 

    #         grx = qtp.Qobj(tensorl([rx(np.pi / 2, noise=self.noise) for i in range(N)]), dims = [[2] * N, [2] * N], shape = (2 ** N, 2 ** N), type = 'oper', isherm = True)
    #         psirhoy = grx * psirho * grx.dag()
    #         prob = psirhoy.diag()
    #         prob = generateRandomProb_s(prob, stats = stats)
    #         eyy = np.dot(prob, Basezz) 

    #         if self.Htype == "XY":
    #             Hexpect = self.h*ez + exx + eyy
    #         elif self.Htype == "XXZ":
    #             Hexpect = self.h*ezz + exx + eyy

    #     return  np.real(Hexpect)

    # def grad_parashift_XY(self, thatal, alpha, samples):
    #     grad_theta = []
    #     dth = np.zeros(len(thetal))
    #     for i in range(len(thetal)):
    #         if not (i % 4 == 2):
    #             ###parameter shift for single-qubit gate
    #             dth[i] = np.pi/2
    #             loss1 = self.loss_quantum(thetal + dth, alpha, samples)
    #             dth[i] = -np.pi/2
    #             loss1 -= self.loss_quantum(thetal + dth, alpha, samples)
    #             grad_theta.append(loss1*0.5)
    #             dth[i] = 0

    #         else:
    #             ####parameter shift for XY gate
    #             lossX = self.loss_quantum_grad(thetal, alpha, samples, i, np.pi/8, np.pi/8) - self.loss_quantum_grad(thetal, alpha, samples, i, -np.pi/8, -np.pi/8)
                
    #             lossY = self.loss_quantum_grad(thetal, alpha, samples, i, np.pi/ 8, -np.pi/8) - self.loss_quantum_grad(thetal, alpha, samples, i, -np.pi/8, np.pi/8)
    #             grad_theta.append(lossX + lossY)
        
    #     return np.array(grad_theta)


if __name__ == "__main__":
    import sys
    def main(beta, h, delta, ti, plot):
        plot=int(plot)
        N = config.N
        beta = float(beta)
        h = float(h)
        delta = float(delta)
        ti = int(ti)
        layer_number = config.layer_number
        trialnum = 4
        decay_step = 20 
        #---------------Optimize-------------------
        bVQE = betaVQE(N, beta, h, layer_number, delta=delta, lr = 0.1, Htype = "XXZ", samp=True, spsa_gradient=True, join=False,  state2=False, noise=False, decay=0.05, niter=250,
        decay_step=decay_step, savedir="data_shadow_circXXZ/" #ndata_%d/" %decay_step
        )
        if not plot:
            bVQE.learn(ti=ti)
        #plot=True
        F_exact, E_exact, S_exact = np.array(exact(bVQE.beta, (bVQE.H).H))

        if plot:
            #----------------Plot results------W---------------
            tag = bVQE.create_tag()
            fig, ax = plt.subplots()
            for ti in range(trialnum):
                tagi = bVQE.savedir + tag + "_ti" + str(ti)
                opt_traj = np.load(tagi + ".npy")
                if bVQE.samp:
                    tag_loss32 = bVQE.savedir + "loss32_" + tag + "_ti" + str(ti) 
                    loss32 = np.load(tag_loss32 + ".npy")
                    ax.plot(np.arange(0, len(loss32)), loss32, label="ti %d" %ti)

                ax.plot(np.arange(0, len(opt_traj)), opt_traj, label="ti %d" %ti) 
                ax.plot([0, len(opt_traj)], [F_exact, F_exact], '--')
                ax.set_xlabel("budget", fontsize = 12)
                ax.set_ylabel("loss function", fontsize = 12)
                ax.text(0, 1.05, tag, wrap = True, transform = ax.transAxes, fontsize = 8)

            ax.legend()
            if not os.path.exists("expresults"):
                    os.mkdir("expresults")
            tagfig = "expresults/" + tag + ".png"
            fig.savefig(tagfig, dpi = 400)
            plt.show()
        return 0

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])




    

