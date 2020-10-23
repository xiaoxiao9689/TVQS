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


def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper

def create_tag(Htype, N, beta, h, layer_number, niter, nbatch, gate, method, samp, noise, lr, decay):
    tag = "Hamil-" + str(Htype)
    tag += "_N" + str(N)
    tag += "_beta" + str(beta)
    tag += "_h" + str(h)
    tag += "_layer_number" + str(layer_number)
    tag += "_method-" + str(method)
    tag += "_gate-" + str(gate)
    tag += "_noise-" + str(noise)
    if samp:
        tag += "_nbatch" + str(nbatch)

    if method == "adam":
        tag += "_lr" + str(lr)
        tag += "_decay" + str(decay)

    return tag


N = config.N
beta = 0.3
h = 0.5
layer_number = config.layer_number
eps = 1e-8
niter = 150
nbatch = [50]*niter
H = hamil_XY(h, N)

Htype = "XY"
gate = "chainXY"
method = "zoopt"
samp = False
join = False
noise= False
decay = 0.05
lr = 0.1

# print(H)
##Exact value###

F = exact(beta, H)[0]
E = exact(beta, H)[1]

print(F)
print(E)




####Optimize#######

def learn(args,  gate=gate, method=method, lr=lr, samp=samp, join=join, noise=noise):
    niter, layer_number, beta, H, N,  h, nbatch, ti = args
    if method == "adam":
        para_opt, phi_opt, theta_opt, lossfl = optimize_adam(niter, layer_number, beta, H, N,  h, nbatch, gate=gate, lr = lr, samp=samp, join=join, decay=decay, noise=noise)
    
    elif method == "zoopt":
        para_opt, phi_opt, theta_opt, lossfl = optimize_zoopt(niter, layer_number, beta, H, N, h, nbatch, gate=gate, samp=samp, join=join, noise=noise)

    elif method == "gpso":
        para_opt, phi_opt, theta_opt, lossfl = optimize_gpso(niter, layer_number, beta, H, N, h, nbatch, gate=gate, samp=samp, join=join, noise=noise)

    else:
        para_opt, phi_opt, theta_opt, lossfl = optimize_scip(niter, layer_number, beta, H, N,  h, nbatch, gate=gate, samp=samp, method=method, join=join, noise=noise)

    
    if not os.path.exists("data"):
        os.mkdir("data")

    tag = create_tag(Htype, N, beta, h, layer_number, niter, nbatch, gate, method, samp, noise, lr, decay)
    tag = "data/" + tag + "_ti" + str(ti)
    np.save(tag, lossfl)

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


# thetal = np.linspace(0, 2*np.pi, (N+1)*layer_number)
# para = np.concatenate(([0.5]*N, thetal))
# samples = gen_btsl(N)
# def func():  
#     #U = get_U(thetal, layer_number, N, gate, noise)
#     loss_func_bound(para, samples, beta, H, N, layer_number, gate=gate, noise=False, samp=samp, join=join)

# start = timeit.default_timer()
# func()
# end = timeit.default_timer()
# print("time: ", end-start, 's')

# outfile = "lossfunc.out"
# profile.run("func()", filename=outfile)
# p = pstats.Stats(outfile)
# p.strip_dirs().sort_stats("cumulative", 'name').print_stats(30)

learn_p = partial(learn, gate=gate, method=method, lr=lr, samp=samp, join=join, noise=noise)


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

# learn([niter, layer_number, beta, H, N,  h, nbatch, 0], gate=gate, method=method, lr=lr, samp=samp, join=join, noise=noise)
##singlecore

# for ti in range(4):
#     learn_p([niter, layer_number, beta, H, N,  h, nbatch, ti])

fig, ax = plt.subplots()
tag = create_tag(Htype, N, beta, h, layer_number, niter, nbatch, gate, method, samp, noise, lr, decay)
for ti in range(4):
    tagi = "data/" + tag + "_ti" + str(ti)
    lossfl = np.load(tagi + ".npy")
    ax.plot(np.arange(0, len(lossfl))[:], lossfl[:]/beta)
    ax.plot([0, len(lossfl)], [F, F], '--')
    ax.set_xlabel("budget", fontsize = 12)
    ax.set_ylabel("loss function", fontsize = 12)
    ax.text(-0.1, 1.05, tag, wrap = True, transform = ax.transAxes)

if not os.path.exists("results"):
        os.mkdir("results")
tagfig = "results/" + tag + ".pdf" 
fig.savefig(tagfig, dpi = 400)
