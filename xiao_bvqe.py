import numpy as np
from func import *
import matplotlib.pyplot as plt
import multiprocessing
import os
import timeit
from functools import partial
from zoopt import Dimension, Objective, Parameter, Opt
def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper
#try
N = 4
beta = 0.3
h = 0.5
layer_number = 4
eps = 1e-8
niter = 200
nbatch = [50]*niter

#Methods:
Htype = "XY"
gate = "XY"
singlegate = "Z"
method = "zoopt_mod"

ps = True
samp = False
join = False
noise= False
decay = 0.05
alpha = 0.1

# set H
if Htype == "XY":
    H = hamil_XY(h, N)
if Htype == "Ising":
    H = hamil_Ising(h, N)
##Exact value###
F = exact(beta, H)[0]
E = exact(beta, H)[1]

print(F)
print(E)





####Optimize#######

def learn(args, singlegate = singlegate, gate=gate, method=method, lr=alpha, samp=samp, join=join, noise=noise, Htype=Htype):
    niter, layer_number, beta, H, N,  h, nbatch, ti = args
    if method == "adam_mod":
        para_opt, phi_opt, theta_opt, lossfl = optimize_adam(niter, layer_number, beta, H, N, singlegate,  h, nbatch, Htype, gate, lr = alpha, samp=samp, join=join, decay=decay, noise=noise)

    elif method == "scip_mod":
        para_opt, phi_opt, theta_opt, lossfl = optimize_scip(niter, layer_number, beta, H, N,  h, nbatch, gate=gate, samp=samp, method=method, join=join, noise=noise)

    elif method == "zoopt_mod":
        para_opt, phi_opt, theta_opt, lossfl = optimize_zoopt(niter, layer_number, beta, H, N, singlegate,  h, nbatch, Htype, gate, lr = alpha, samp=samp, join=join, decay=decay, noise=noise)

    # np.save("data_%dbit_samp_%s/%s/lossf_H_%sbeta%.2fh%.2f_niter_%d_layer_%d_nbatch_%d_alpha_%.3f_%s_ps_%s_%d" %(N, samp,  method, Htype, beta, h,  niter, layer_number, nbatch[0], alpha, join, ps, ti),  np.array(lossfl)/beta)

    # np.save("data_%dbit_samp_%s/%s/para_opt_H_%sbeta%.2fh%.2f_niter_%d_layer_%d_nbatch_%d_alpha_%.3f_%s_ps_%s_%d" % (N, samp,  method, Htype, beta, h,  niter, layer_number, nbatch[0], alpha, join,  ps, ti), para_opt)
    
    np.save("currunt_%d" %ti, lossfl)
#single:
    #np.save("currunt", lossfl)


    if samp:
        samples = gen_samples(phi_opt, 1000, N)
    else:
        samples = gen_btsl(N) 
    
    if join:
        alpha_opt = para_opt[:int(2**N)]
        theta_opt = para_opt[int(2**N):]
    else:
        alpha_opt = para_opt[:N] 
        theta_opt = para_opt[N:]
    F_opt = loss_func(para_opt, samples, beta, H, N, singlegate, layer_number, Htype, gate=gate, samp=samp, join=join, noise=noise)/beta
    E_opt = loss_quantum(theta_opt, alpha_opt, samples, beta, H, N, singlegate, layer_number, Htype, gate=gate, samp=samp, join=join, noise=noise)/beta
    print(para_opt)
    print("F: ", F, " ", "F_opt: ", F_opt, "\n")
    print("E: ", E, " ", "E_opt: ", E_opt, "\n")
    print("alpha_opt: ", alpha_opt, "\n")
    print("theta_opt: ", theta_opt, "\n")
# def learn_wrap(args):
#     return learn(*args)

learn_p = partial(learn, singlegate=singlegate, gate=gate, method=method, lr=alpha, samp=samp, join=join, noise=noise, Htype=Htype)

def test(x):
    np.random.seed()
    sr = np.random.rand(5)
    print(sr)
    
#learn_p((niter, layer_number, beta, H, N,  h, nbatch, 0))    
#multicore
pool = multiprocessing.Pool(4)      
start = timeit.default_timer()
parapack = [(niter, layer_number, beta, H, N,  h, nbatch, ti) for ti in range(4)]
pool.map(learn_p, parapack)
pool.map(test, np.arange(0, 4))
pool.close()
pool.join()
end = timeit.default_timer()
print('multi processing time:', str(end-start), 's')
for i in range(4):
    lossfl = np.load("currunt_%d.npy" %i)
    plt.plot(np.arange(0, len(lossfl)), np.array(lossfl)/beta)
    if method == "adam_mod":
        plt.plot([0, niter], [F, F], '--')
    if method == "zoopt_mod":
        plt.plot([0, len(lossfl)], [F, F], '--')
        
plt.savefig("currunt.png")




#singlecore
#learn([niter, layer_number, beta, H, N,  h, nbatch, 0], gate=gate, method=method, lr=alpha, samp=samp, join=join, noise=noise)
# learn_p([niter, layer_number, beta, H, N,  h, nbatch, 0])
# lossfl = np.load("currunt.npy")
# plt.plot(np.arange(0, len(lossfl)), np.array(lossfl)/beta)
# if method == "adam_mod":
#     plt.plot([0, niter], [F, F], '--')
# if method == "zoopt_mod":
#     plt.plot([0, len(lossfl)], [F, F], '--')
# plt.savefig("currunt.png")

