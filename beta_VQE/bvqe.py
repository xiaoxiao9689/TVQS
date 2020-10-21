import numpy as np
from func import *
import matplotlib.pyplot as plt
import multiprocessing
import os
import timeit
from functools import partial
import profile
import pstats

def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper

N = 6
beta = 0.3
h = 0.5
layer_number = 6
eps = 1e-8
niter = 150
nbatch = [50]*niter
H = hamil_XY(h, N)

Htype = "XY"
gate = "chainXY"
method = "zoopt"
samp = False
join = False
noise= True
decay = 0.05
alpha = 0.1

# print(H)
##Exact value###

F = exact(beta, H)[0]
E = exact(beta, H)[1]

print(F)
print(E)




####Optimize#######

def learn(args,  gate=gate, method=method, lr=alpha, samp=samp, join=join, noise=noise):
    niter, layer_number, beta, H, N,  h, nbatch, ti = args
    if method == "adam":
        para_opt, phi_opt, theta_opt, lossfl = optimize_adam(niter, layer_number, beta, H, N,  h, nbatch, gate=gate, lr = alpha, samp=samp, join=join, decay=decay, noise=noise)
    
    elif method == "zoopt":
        para_opt, phi_opt, theta_opt, lossfl = optimize_zoopt(niter, layer_number, beta, H, N, h, nbatch, gate=gate, samp=samp, join=join, noise=noise)

    elif method == "gpso":
        para_opt, phi_opt, theta_opt, lossfl = optimize_gpso(niter, layer_number, beta, H, N, h, nbatch, gate=gate, samp=samp, join=join, noise=noise)

    else:
        para_opt, phi_opt, theta_opt, lossfl = optimize_scip(niter, layer_number, beta, H, N,  h, nbatch, gate=gate, samp=samp, method=method, join=join, noise=noise)

    # np.save("data_%dbit_samp_%s/%s/lossf_H_%sbeta%.2fh%.2f_niter_%d_layer_%d_nbatch_%d_alpha_%.3f_%s_ps_%s_%d" %(N, samp,  method, Htype, beta, h,  niter, layer_number, nbatch[0], alpha, join, ps, ti),  np.array(lossfl)/beta)

    # np.save("data_%dbit_samp_%s/%s/para_opt_H_%sbeta%.2fh%.2f_niter_%d_layer_%d_nbatch_%d_alpha_%.3f_%s_ps_%s_%d" % (N, samp,  method, Htype, beta, h,  niter, layer_number, nbatch[0], alpha, join,  ps, ti), para_opt)
    
    np.save("current_%d" %ti, lossfl)



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


    if method == "zoopt" or method == "gpso": 
        F_opt = loss_func_bound(para_opt, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=noise)/beta

        alpha_opt = -np.log(1.0/para_opt[:N]-1)
    else:
        F_opt = loss_func(para_opt, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=noise)

    E_opt = loss_quantum(theta_opt, alpha_opt, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=noise)/beta
    print(para_opt)
    print("F: ", F, " ", "F_opt: ", F_opt, "\n")
    print("E: ", E, " ", "E_opt: ", E_opt, "\n")

# def learn_wrap(args):
#     return learn(*args)

learn_p = partial(learn, gate=gate, method=method, lr=alpha, samp=samp, join=join, noise=noise)

def test(x):
    np.random.seed()
    sr = np.random.rand(5)
    print(sr)


# thetal = np.linspace(0, 2*np.pi, (N+1)*layer_number)
# para = np.concatenate(([0.5]*N, thetal))
# samples = gen_btsl(N)
# def func():  
#     #get_U(thetal, layer_number, N, gate, noise)
#     loss_func_zoopt(para, samples, beta, H, N, layer_number, gate=gate, noise=noise, samp=samp, join=join)

# start = timeit.default_timer()
# func()
# end = timeit.default_timer()
# print("time: ", end-start, 's')

# outfile = "lossfunc.out"
# profile.run("func()", filename=outfile)
# p = pstats.Stats(outfile)
# p.strip_dirs().sort_stats("cumulative", 'name').print_stats(30)



####multicore
# pool = multiprocessing.Pool(4)      
# start = timeit.default_timer()
# parapack = [(niter, layer_number, beta, H, N,  h, nbatch, ti) for ti in range(4)]
# pool.map(learn_p, parapack)
# # pool.map(test, np.arange(0, 4))
# pool.close()
# pool.join()
# end = timeit.default_timer()
# print('multi processing time:', str(end-start), 's')

##singlecore
# learn([niter, layer_number, beta, H, N,  h, nbatch, 0], gate=gate, method=method, lr=alpha, samp=samp, join=join, noise=noise)
for i in range(4):
    learn_p([niter, layer_number, beta, H, N,  h, nbatch, i])

for i in range(4):
    lossfl = np.load("current_%d.npy" %i)
    plt.plot(np.arange(0, len(lossfl))[:], lossfl[:]/beta)
    plt.plot([0, len(lossfl)], [F, F], '--')
plt.savefig("current.pdf")
