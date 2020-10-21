from func import *
from error_gate import *
import time
#test for q_expect###

#Global control
SAMP = False
JOIN = False


len_theta_XY = 0.1
len_phi_XY = 0.2



def hamil_z(N):
    sx_list, sy_list, sz_list = Nbit_single(N)
    H = 0
    for i in range(N):
        H += sz_list[i]
    return H

def hamil_zz(N):
    sx_list, sy_list, sz_list = Nbit_single(N)
    H = 0
    for i in range(N-1):
        H += np.dot(sz_list[i], sz_list[i+1])
    return H


def test_q_expect():
    h = 0.5
    N = 4
    beta = 0.3
    H = hamil_XY(h, N)
    layer_number = 4
    noise=  True
    #thetal = 2*np.pi*np.random.rand((layer_number*(N-1)*4))
    thetal = np.linspace(0, 2*np.pi, (layer_number*(N-1)*4))

    U = get_U(thetal, layer_number, N, "XY", False)
    U_error = get_U(thetal, layer_number, N, "XY", noise)

    x = np.array([1, 0, 1, 0])


    start1 = time.time()
    e1 = q_expect(U, x, beta, H)
    end1 = time.time()
    print("E1: ", e1)
    print("E1 time: ", end1-start1)

    basez = get_baseglobalz(N)
    basezz = get_baseglobalzz(N)

    start2 = time.time()
    e2 = q_expect_exp(U_error, x, beta, h, basez, basezz, noise)
    end2 = time.time()

    print("E2: ", e2)
    print("E2 time: ", end2-start2)



def test_loss():
    h = 0.5
    N = 4
    beta = 0.3
    H = hamil_XY(h, N)
    layer_number = 4
    noise=  True
    alpha = np.array([1., 0, 2,  -0.7])
    #thetal = 2*np.pi*np.random.rand((layer_number*(N-1)*4))
    thetal = np.linspace(0, 2*np.pi, (layer_number*(N-1)*4))
    para = np.concatenate((alpha, thetal))
    gate = "XY"
    samples = gen_btsl(N)
    samp = False
    join = False


    lqexact = loss_quantum(thetal, alpha, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=False)
    
    lqerror = []
    for i in range(50):
        lqerror.append(loss_quantum(thetal, alpha, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=True))
    


    lfexact = loss_func(para, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=False)

    lferror = []
    for i in range(1):
        lferror.append(loss_func(para, samples, beta, H, N, layer_number, gate=gate, samp=samp, join=join, noise=True))

    plt.plot([0, 50], [lqexact, lqexact], '--', label="exact")
    plt.plot(np.arange(0, 50), lqerror, 'o', mfc='none', color="red", label="fide=0.98")
    plt.plot([0, 50], [np.mean(lqerror)]*2,  color='red')
    plt.ylim(-0.4, 0.4)
    plt.xlabel("times")
    plt.ylabel("loss_quantum")
    plt.legend()
    #plt.title("fidelity=0.999")
    # ax[1].plot([0, 50], [lfexact, lfexact], '--', label="")
    # ax[1].plot(np.arange(0, 50), lferror, 'o', mfc = 'none')
    # ax[1].plot([0, 50], [np.mean(lferror)]*2)
    plt.tight_layout()
    # plt.savefig("fid0.98.png")

test_loss()


