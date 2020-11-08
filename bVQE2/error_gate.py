import numpy as np
from scipy.linalg import expm
from functools import reduce
import qutip as qtp
import matplotlib.pyplot as plt
from util import si, sx, sy, sz

def noise_rg(theta, dtheta, dphi, gate):
    if gate == "rx":
        nx = np.sin(np.pi / 2) * np.cos(dphi)
        ny = np.sin(np.pi / 2) * np.sin(dphi)
        nz = np.cos(np.pi / 2)
        #rg = expm(-1j*(theta+dtheta)/2*(nx*sx+ny*sy+nz*sz))
        rg = np.cos((theta + dtheta) / 2) * si - 1j * np.sin((theta + dtheta) / 2) * (nx * sx + ny * sy + nz * sz)  


    elif gate == "ry":
        nx = np.sin(np.pi / 2) * np.cos(np.pi / 2 + dphi)
        ny = np.sin(np.pi / 2) * np.sin(np.pi / 2 + dphi)
        nz = np.cos(np.pi / 2)
        #rg = expm(-1j*(theta+dtheta)/2*(nx*sx+ny*sy+nz*sz))
        rg = np.cos((theta + dtheta) / 2) * si - 1j * np.sin((theta + dtheta) / 2) * (nx * sx + ny * sy + nz * sz)  
    
    elif gate == "rxy":
        rg = expm(-1j*((theta+dtheta)*(np.kron(sx, sx)+np.kron(sy, sy))+dphi*np.kron(sz, si)))    
    
    return rg



def rand_rx(theta):
    len_theta = 0.05    
    len_phi = 0.05
    dtheta = np.random.uniform(-len_theta, len_theta)
    dphim = np.sqrt(len_phi ** 2 - (len_phi * dtheta) ** 2 / len_theta ** 2)
    dphi = np.random.uniform(-dphim, dphim)
    noise_rx = noise_rg(theta, dtheta, dphi, 'rx')

    # ##check fidelity
    # idrg = qtp.rx(theta)
    # fid = fidelity(idrg, noise_rx)
    # print(fid)

    return noise_rx

def rand_ry(theta):
    len_theta = 0.05
    len_phi = 0.05
    dtheta = np.random.uniform(-len_theta, len_theta)
    dphim = np.sqrt(len_phi ** 2 - (len_phi * dtheta) ** 2 / len_theta ** 2)
    dphi = np.random.uniform(-dphim, dphim)
    noise_ry = noise_rg(theta, dtheta, dphi, 'ry')

    ##check fidelity
    # idrg = qtp.ry(theta)
    # fid = fidelity(idrg, noise_ry)
    # print(fid)
    
    return noise_ry

def rand_xy(theta, dthedphi=None):
    len_theta = 0.1
    len_phi = 0.2
    if dthedphi == None:
        dtheta = (np.random.rand()-0.5)/0.5*len_theta
        dphim = np.sqrt(len_phi**2 - (len_phi*dtheta)**2/len_theta**2)
        dphi = (np.random.rand()-0.5)/0.5*dphim
    else:
        dtheta, dphi = dthedphi
    noise_rxy = noise_rg(theta, dtheta, dphi, 'rxy')

    #check fidelity
    # idrg = (-1j*theta*(qtp.tensor(qtp.sigmax(), qtp.sigmax()) + qtp.tensor(qtp.sigmay(), qtp.sigmay()))).expm()
    # fid = fidelity(idrg, noise_rxy)
    # print(dtheta, dphi)
    # print(fid)

    return noise_rxy


def rx(phi, noise):
    if noise:
        return rand_rx(phi)
    else:
        return np.array([[np.cos(phi / 2), -1j * np.sin(phi / 2)],
                     [-1j * np.sin(phi / 2), np.cos(phi / 2)]])
            
def ry(phi, noise):
    if noise:
        return rand_ry(phi)
    else:
        return np.array([[np.cos(phi / 2), -np.sin(phi / 2)],
                     [np.sin(phi / 2), np.cos(phi / 2)]])

def rz(phi, noise):
    if noise:
        #Use rx and ry to construct rz
        return reduce(np.dot, [rand_rx(-np.pi/2), rand_ry(phi), rand_rx(np.pi/2)])
    else:
        return np.array([[np.exp(-1j * phi / 2), 0],
                     [0, np.exp(1j * phi / 2)]])

def qN_rz(phi, noise):
    ''' Error rz for simulation with symmetry, dphi = 0.05 for fidelity 0.999;
    0.1 for fidelity 0.995 '''

    if noise:
        dphi = np.random.uniform(-0.05, 0.05)
        return np.array([[np.exp(-1j * (phi + dphi) / 2), 0],
                        [0, np.exp(1j * (phi + dphi) / 2)]])
    else:
        return np.array([[np.exp(-1j * phi / 2), 0],
                        [0, np.exp(1j * phi / 2)]])

def rxy(theta, noise, dthedphi=None):
    if noise:
        return rand_xy(theta, dthedphi=dthedphi)
    else:
        return expm(-1j*theta*(np.kron(sx, sx) + np.kron(sy, sy)))



def fidelity(idg, g):
    Nbit = len(idg.dims[0])
    g = qtp.Qobj(g, dims=[[2]*Nbit, [2]*Nbit], shape=(2**Nbit, 2**Nbit))
    idg_rho = qtp.spre(idg)*qtp.spost(idg.dag())
    g_rho = qtp.spre(g)*qtp.spost(g.dag())
    op_basis = [[qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()] for i in range(Nbit)]
    chi_idg = qtp.qpt(idg_rho, op_basis)
    chi_g = qtp.qpt(g_rho, op_basis)
    fid = np.trace(np.dot(chi_idg, np.conj(chi_g).T))/(np.sqrt(np.trace(np.dot(chi_idg, np.conj(chi_idg).T)))*np.sqrt(np.trace(np.dot(chi_g, np.conj(chi_g).T))))
    
    return np.abs(fid)


#### test for fidelity#####
# theta = np.pi/2
# idrx = qtp.rx(theta)
# noise_rx = noise_r(theta, 0.01, 0.01, 'rx')
#fid = fidelity(idrx, noise_rx)
#print(fid)


def cal_fidel(theta, dthetam, dphim, gate):
    if gate == "rx":
        idrg = qtp.rx(theta)
    elif gate == "ry":
        idrg = qtp.ry(theta)
    elif gate == 'rxy':
        idrg = (-1j*theta*(qtp.tensor(qtp.sigmax(), qtp.sigmax()) + qtp.tensor(qtp.sigmay(), qtp.sigmay()))).expm()  

    dthetal = np.arange(-dthetam, dthetam, 0.01)
    dphil = np.arange(-dphim, dphim, 0.01)
    fidel = np.zeros((len(dthetal), len(dphil)))
    for i in range(len(dthetal)):
        for j in range(len(dphil)):
            noiserg = noise_rg(theta, dthetal[i], dphil[j], gate)
            fidel[i, j] = fidelity(idrg, noiserg)
    
    return fidel

def find_circ(theta, gate, fide):
    dthetam = 0.3
    dphim = 0.3
    dthetal = np.arange(-dthetam , dthetam, 0.01)
    dphil = np.arange(-dphim , dphim , 0.01)
    
    fidel = cal_fidel(theta, dthetam, dphim , gate)
    contour = plt.contour(dthetal, dphil, fidel, [fide], color='k')
    p = contour.collections[0].get_paths()[0]
    plt.xlabel("dphi")
    plt.ylabel("dtheta")
    plt.title("fidd=%.3f" %fide)
    par = p.vertices

    len_x = np.max(par[:, 0])
    len_y = np.max(par[:, 1])

    #dTh, dPh = np.meshgrid(dthetal, dphil) 
    #z = dTh**2/(len_x**2)+dPh**2/(len_y)**2-1
    #plt.contour(dthetal,  dphil, z, 0, color='b')
    # fig, ax = plt.subplots()
    # im = ax.imshow(fidel)
    # fig.colorbar(im)

    return [len_x, len_y]




if __name__ == '__main__':
    gate = 'rxy'
    fide = 0.999
    #thetal = np.linspace(0, 2*np.pi, 50)
    thetal = [np.pi/2]
    ellipl = []
    for theta in thetal:
        ellip = find_circ(theta, gate, fide)
        print(ellip)
        ellipl.append(ellip)
    
    #np.save("ellip_%.3f_%s" %(fide, gate), ellipl)



    
    
    
