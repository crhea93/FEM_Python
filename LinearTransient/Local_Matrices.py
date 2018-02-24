
import numpy as np
from elements import Element
from B_mat import  get_B_plane_stress

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
def getMassMatrix(C):
    E = Element("Q4")
    weights = [1,1]
    xIP=[-np.sqrt(3)/3,np.sqrt(3)/3]
    Me = np.zeros((8,8))
    me(xIP, C, Me,E)
    return Me

def me(xip,C,Me,E):
    for i in range(len(xip)):
        for j in range(len(xip)):
            N = E.N(xip[i],xip[j])
            N_T = N.transpose()
            Gradn,detJ = E.G_map(xip[i],xip[j],C)
            Me +=  np.matmul(N_T,N)*detJ
