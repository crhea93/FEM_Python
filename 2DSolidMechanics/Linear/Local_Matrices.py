
import numpy as np
from elements import Element
from B_mat import  get_B_plane_stress
from multiprocessing import Pool as ThreadPool
import threading

def Dif_basic_K(Coords,k=1): #let k be an optional arguement
    E = Element("Q4")

    Ke = np.zeros((len(Coords),len(Coords)))

    x_ip = [-np.sqrt(3)/3,np.sqrt(3)/3]

    W_ip = [1,1]
    for i in range(len(x_ip)):
        for j in range(len(x_ip)): # now ready to integrate
            G_m = E.G_map()
            B,Jdet = G_m(x_ip[i],x_ip[j],Coords)
            B_T = np.transpose(B)
            Ke = Ke + B_T.dot(B)*k* Jdet * W_ip[i] * W_ip[j]

    return Ke


def getStiffnessmatrix(C,D,thickness,el_epsilon):
    E = Element("Q4")
    weights = [1,1]
    xIP=[-np.sqrt(3)/3,np.sqrt(3)/3]
    Ke = np.zeros((8,8))
    ke(xIP, C, D, el_epsilon,  weights, thickness,E,Ke)
    return Ke

def ke(xIP,C,D,el_epsilon,weights,thickness,E,Ke):
    for i in range(len(xIP)):
        for j in range(len(xIP)):
            gradN, detJ = E.G_map(xIP[i], xIP[j], C)
            B = get_B_plane_stress(gradN)
            Ke += np.transpose(B) * D(el_epsilon) * B * detJ * weights[i] * weights[j] * thickness

def getInternalForces(Coord_mat_el,sigma):
    E = Element("Q4")
    weights = [1, 1]
    xIP = [-np.sqrt(3) / 3, np.sqrt(3) / 3]
    fint_pre = np.zeros((1,8))
    count = 0
    for i in range(len(xIP)):
        for j in range(len(xIP)):
            gradN, detJ = E.G_map(xIP[i], xIP[j], Coord_mat_el)
            B = get_B_plane_stress(gradN)
            fint_pre += np.dot(np.transpose(B),sigma[:,count]) * detJ * weights[i] * weights[j]
            count += 1

    return np.transpose(fint_pre)
def getdkdumatrix(C,D_prime,thickness,el_sigma):
    E = Element("Q4")
    weights = [1,1]
    xIP=[-np.sqrt(3)/3,np.sqrt(3)/3]
    Ke = np.zeros((8,8))
    ke_prime(xIP, C, D_prime, el_sigma, weights, thickness,E,Ke)

    return Ke

def ke_prime(xIP,C,D_prime,el_epsilon,weights,thickness,E,Ke):
    for i in range(len(xIP)):
        for j in range(len(xIP)):
            gradN, detJ = E.G_map(xIP[i], xIP[j], C)
            B = get_B_plane_stress(gradN)
            Ke += np.transpose(B) * D_prime(el_epsilon) * B * detJ * weights[i] * weights[j] * thickness

'''def applyNeumann(Coord_mat_el,Coord_mat_ind,NBC,trac):
    f_grav = np.zeros((8,1))
    E = Element("Q4")
    weights = [1, 1]
    xIP = [-np.sqrt(3) / 3, np.sqrt(3) / 3]
    for el_in in range(len(Coord_mat_ind)):
        if el_in+1 in NBC: #only integrate if  the value is on the node
            for k in range(len(xIP)):
                s = xIP[k]
                N = E.N(s,s)
                J_s = np.matrix(detJsideQ4(s,s))
                n_mat = np.matrix(N.transpose())
                trac_mat =  np.matrix(trac)
                f_grav += n_mat*trac_mat*J_s[0,k]*weights[k]
	return f_grav'''
def detJsideQ4(x,y):
    G = np.array([-1/2,1/2])
    dxdpsi = G*x
    dydpsi = G*y
    detJside = np.sqrt(dxdpsi**2+dydpsi**2)
    return detJside
