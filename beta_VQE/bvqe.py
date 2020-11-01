import numpy as np
from func import *
import matplotlib.pyplot as plt
import multiprocessing
import os
import timeit
from functools import partial
import profile
import pstats
import config



def create_tag(Htype, N, beta, h, layer_number, niter, nbatch, gate, method, samp, noise, symmetry, lr, decay, peierls, decoherence):
    tag = "Hamil-" + str(Htype)
    tag += "_N" + str(N)
    tag += "_beta" + str(beta)
    tag += "_h" + str(h)
    tag += "_layer_number" + str(layer_number)
    tag += "_method-" + str(method)
    tag += "_gate-" + str(gate)
    tag += "_noise" if noise else ""
    tag += "_sym" if symmetry else ""
    tag += "_peierls" if peierls else ""
    tag += "_decoherence" if decoherence else ""
    if samp:
        tag += "_nbatch" + str(nbatch)

    if method == "adam":
        tag += "_lr" + str(lr)
        tag += "_decay" + str(decay)

    return tag


N = config.N
beta = 0.2
h = 0.5
layer_number = config.layer_number
eps = 1e-8
niter = 150
nbatch = 10
Htype = "XY"
gate = "chainXY"
method = "zoopt"
samp = False
join = False
noise= False 
decay = 0.05
lr = 0.1
symmetry = False
peierls = True
decoherence = False

H = Hamiltonian(h, N, Htype = Htype)

# print(H)
##Exact value###
F = exact(beta, H.H)[0]
E = exact(beta, H.H)[1]

print(F)
print(E)

#Do not use the probability to generate the samples but fix the state |x>, then just do variation of phi.


####Optimize#######

def learn(args,  nbatch = 100, gate=gate, method=method, lr=lr, samp=samp, join=join, noise=noise, symmetry = False, peierls = False, decoherence = False):
    niter, layer_number, beta, H, nbatch, ti = args
    
    var_basis = gen_btsl(N)
    #var_basis = gen_btsl_sub(N, 20) 

    if method == "adam":
        para_opt, phi_opt, theta_opt, lossfl = optimize_adam(niter, layer_number, beta, H, var_basis, nbatch = nbatch, gate=gate, lr = lr, samp=samp, join=join, decay=decay, noise=noise, peierls = peierls)
    
    elif method == "zoopt":
        para_opt, phi_opt, theta_opt, lossfl = optimize_zoopt(niter, layer_number, beta, H,  var_basis, nbatch = nbatch, gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry, peierls = peierls, decoherence = decoherence)

    elif method == "gpso":
        para_opt, phi_opt, theta_opt, lossfl = optimize_gpso(niter, layer_number, beta, H,  var_basis,  nbatch = nbatch, gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry)
    elif method == "direct":
        para_opt = optimize_direct(niter, layer_number, beta, H,  var_basis,  nbatch = nbatch, gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry)

    else:
        para_opt, phi_opt, theta_opt, lossfl = optimize_scip(niter, layer_number, beta, H, var_basis, nbatch = nbatch, gate=gate, samp=samp, method=method, join=join, noise=noise)

    
    if not os.path.exists("data"):
        os.mkdir("data")

    tag = create_tag(Htype, N, beta, h, layer_number, niter, nbatch, gate, method, samp, noise, symmetry, lr, decay, peierls, decoherence)
    tagloss = "data/" + tag + "_ti" + str(ti)
    print(tagloss)
    tagpara = "data/" + "para" + tag + "_ti" + str(ti)

    np.save(tagloss, lossfl)
    np.save(tagpara, para_opt)
        
    
    if join:
        alpha_opt = para_opt[:len(var_basis)]
        theta_opt = para_opt[len(var_basis):]
    else:
        alpha_opt = para_opt[:N] 
        theta_opt = para_opt[N:]
    
    if samp:
        var_basis = gen_samples(phi_opt, 1000, N)

    #var_basis = gen_btsl(N)
    ###Peierls
    if peierls:
        theta_opt = para_opt

    if method == "zoopt" or method == "gpso":
        if peierls:  
            F_opt = loss_func_peierls(para_opt, var_basis, beta, H, layer_number, nbatch = 1000, gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry, decoherence = decoherence)/beta
        else:
            F_opt = loss_func_bound(para_opt, var_basis, beta, H, layer_number,  gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry, decoherence = decoherence)/beta
 
        alpha_opt = -np.log(1.0/para_opt[:N]-1)
    else:
        F_opt = loss_func(para_opt, var_basis, beta, H, layer_number, gate=gate, samp=samp, nbatch = 1000,  join=join, noise=noise, symmetry = symmetry)/beta

    E_opt = loss_quantum(theta_opt, alpha_opt, var_basis, beta, H, layer_number,  gate=gate, samp=samp, join=join, noise=noise, symmetry = symmetry, peierls = peierls,decoherence = decoherence)
    print(para_opt)
    print("F: ", F, " ", "F_opt: ", F_opt, "\n")
    print("E: ", E, " ", "E_opt: ", E_opt, "\n")


learn_p = partial(learn, gate=gate, method=method, lr=lr, samp=samp, join=join, noise=noise, symmetry = symmetry, peierls = peierls, decoherence = decoherence)

if method == "adam" or method == "lbfgs" or method == "gpso":
    #parallel
    pool = multiprocessing.Pool(4)      
    start = timeit.default_timer()
    parapack = [(niter, layer_number, beta, H, nbatch, ti) for ti in range(4)]
    pool.map(learn_p, parapack)
    # pool.map(test, np.arange(0, 4))
    pool.close()
    pool.join()
    end = timeit.default_timer()
    print('multi processing time:', str(end-start), 's')


elif method == "zoopt" or method == "direct":
    ##already parallel in zoopt
    for ti in range(4):
        learn_p([niter, layer_number, beta, H, nbatch, ti])


#Plot results
fig, ax = plt.subplots()
tag = create_tag(Htype, N, beta, h, layer_number, niter, nbatch, gate, method, samp, noise, symmetry, lr, decay, peierls, decoherence)
print(tag)
for ti in range(4):
    tagi = "data/" + tag + "_ti" + str(ti)
    lossfl = np.load(tagi + ".npy")
    ax.plot(np.arange(0, len(lossfl))[:], lossfl[:]/beta)
    ax.plot([0, len(lossfl)], [F, F], '--')
    ax.set_xlabel("budget", fontsize = 12)
    ax.set_ylabel("loss function", fontsize = 12)
    ax.text(0, 1.05, tag, wrap = True, transform = ax.transAxes, fontsize = 6)

if not os.path.exists("results"):
        os.mkdir("results")
tagfig = "results/" + tag + ".pdf" 
fig.savefig(tagfig, dpi = 400)
