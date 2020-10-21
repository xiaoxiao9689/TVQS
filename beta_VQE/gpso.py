# Global particle swarm optimization

import numpy as np
import matplotlib.pyplot as plt
#from zoopt import Dimension, Objective, Parameter, Opt, Solution
import DIRECT


def Rastrgin(x):
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)

def Ackley(x, a = 20, b = 0.2, c =  2*np.pi):
    y = - a*np.exp( - b*np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(c*x))) + a + np.exp(1)

    return y 
    

def Rosenbrock(x):
    return np.sum([100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1)])

def minimize(func, args = (), max_iter = 1000, dim = 0, bound = np.array([]), decay = False, popsize = 20, wmin = 0.4,  wmax = 0.5,  c1 = 1.0, c2 = 1.0, c3 = 1.0, vp = 0.1, boundary_handling = None):
    lowbound = np.repeat([bound[:, 0]], popsize,  axis = 0)
    upbound = np.repeat([bound[:, 1]], popsize, axis = 0)

    #initial population
    xpop = np.array([np.random.uniform(bound[:, 0], bound[:, 1]) for i in range(popsize)])
    vbound = bound*vp
    vpop = np.array([np.random.uniform(vbound[:, 0], vbound[:, 1]) for i in range(popsize)])
    

    valuepop = np.array([func(xpop[i], *args) for i in range(popsize)])
    valuePbest = valuepop
    valueGbest = np.min(valuePbest)
    Pbest = xpop
    Gbest = np.zeros((max_iter, dim))
    Gbest[0] = Pbest[np.argmin(valuePbest)]
    traj = np.zeros(max_iter)
    traj[0] = valueGbest
    for j in range(1, max_iter):
        #choose Ebest    
        ch_iter = int(np.random.rand()*j)
        ch_dim = int(np.random.rand()*dim)
        Ebest = Gbest[ch_iter, ch_dim]

        w = wmax
        if decay:
            w = wmax - (wmax - wmin) * (j / max_iter)

        #update v
        vpop = w*np.random.rand()*vpop + c1*(1 + np.random.rand())*(Pbest - xpop) + c2*(np.random.rand())*(Gbest[j-1] - xpop) + c3*np.random.rand()*(Ebest - xpop)
        np.clip(vpop, vbound[:, 0], vbound[:, 1])

        #update x
        xpop = xpop + vpop

        #boundary_handling
        if boundary_handling == "random":
            ind = xpop > upbound
            xpop[ind] = np.random.uniform(lowbound[ind], upbound[ind])
            ind = xpop < lowbound
            xpop[ind] = np.random.uniform(lowbound[ind], upbound[ind])

        elif boundary_handling == "reflect":
            ind = xpop > upbound
            vpop[ind] = -vpop[ind]
            ind = xpop < lowbound
            vpop[ind] = -vpop[ind]
            xpop = np.clip(xpop, bound[:, 0], bound[:, 1])
        
        elif boundary_handling == "periodic":
            ind = xpop > upbound
            xpop[ind] = lowbound[ind] + (xpop[ind]-upbound[ind])%(upbound[ind] - lowbound[ind])
            ind = xpop < lowbound
            xpop[ind] = upbound[ind] - (lowbound[ind] - xpop[ind])%(upbound[ind] - lowbound[ind])

        elif boundary_handling == "absorb":
            #absorb
            pass
        else:
            xpop = np.clip(xpop, bound[:, 0], bound[:, 1])
        
        #Perform variation
        # p = np.random.rand()
        # if p > 0.8:
        #     k = np.random.randint(0, popsize)
        #     xpop[k] = np.random.uniform(bound[:, 0], bound[:, 1])


        valuepop = np.array([func(xpop[i], *args) for i in range(popsize)])

        ind = valuepop < valuePbest
        valuePbest[ind] = valuepop[ind]
        Pbest[ind] = xpop[ind]

        ##Perform mutation
        # K = 0.8
        # U = K * np.mean(valuePbest)
        # goodind = valuePbest < U

        # if len(valuePbest[goodind]) > 0:
        #     randind = int(np.random.rand()*len(valuePbest[goodind]))
        #     Pbest1 = Pbest[goodind][randind]
        # #Pbest1 = Pbest[randind]

        #     y = np.random.uniform(bound[:, 0], bound[:, 1])
        #     trail = (1 + 0.5)*Pbest1 - 0.5*y
        #     trailvalue = func(trail, *args)
        #     print("trail", trail)
        #     worstind = np.argmax(valuePbest)
        #     if trailvalue < valuePbest[worstind]:
        #         Pbest[worstind] = trail 
        #         print("huande", Pbest[worstind])

        #update solution
        ind = np.argmin(valuePbest)
        valueGbest = valuePbest[ind]
        Gbest[j] = Pbest[ind]
        traj[j] = valueGbest
        print(valueGbest)
    return Gbest[-1], traj


if __name__ == '__main__':
    #GPSO results
    dim = 20
    bound = np.array([[-5.12, 5.12]]*dim)
    trajx, traj = minimize(Rastrgin, max_iter=1000, dim=dim, bound=bound, wmin = 0.3, wmax = 0.5, decay = False, boundary_handling = "periodic")
    plt.plot(np.arange(len(traj)), traj, label = 'gpso')
    plt.ylim(0)


    # ##zoopt results
    # def objfunc(solution):
    #     x = np.array(solution.get_x())
    #     return Ackley(x)

    # obj = Objective(objfunc, Dimension(dim, bound, [True]*dim))
    # sol = Opt.min(obj, Parameter(budget=4000,  exploration_rate=0.01))
    # para = np.array(sol.get_x())
    # lossfl = obj.get_history_bestsofar()
    # print(len(lossfl))
    # plt.plot(np.arange(len(lossfl)), lossfl, label = 'racos_zoopt')
    # plt.ylim(0)
    # plt.legend()

    




        









        
         






     

